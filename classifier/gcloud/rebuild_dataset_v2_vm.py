"""Runs ON the temporary dataset-rebuild VM (see rebuild_dataset_v2.sh).

Rebuilds two pools directly in the bucket, all audio normalized to 16 kHz mono
WAV (config.py SAMPLE_RATE), all SAA long-paragraph clips split into
GLOBE-like windows so clip length stops being a usable "tell":

  1. curated/<CC>/  (country accent classes, 6: US UK CA AU IN CN)
     - resample everything to 16 kHz mono
     - split every SAA clip (~18-58s paragraphs) into ~0.4-7.5s segments at
       natural pauses (silence-based), matching GLOBE's own duration range
       (0.4-11s, see docs/assets/duration_eda_20260717-000058/summary.csv)
     - US/UK/CA (>5000 after processing): speaker-stratified downsample to
       5000 (SAA-derived segments capped at 3/orig-speaker first, so one long
       paragraph can't dominate the sample)
     - AU/IN/CN: keep everything (best effort; CN is intentionally left
       under-sized per DATASET.md's "floor" caveat, IN lands closer to 5000
       than before thanks to the SAA split but is not forced to it)
     - old curated/ has ALREADY been archived + deleted by the orchestrator
       before this VM starts (rebuild_dataset_v2.sh) -- this script writes a
       clean new curated/<CC>/.

  2. curated_spoof/real_fake_5k/  (flat real-vs-fake pool, exactly 1:1)
     - real  = ASVspoof-2019-LA bonafide (5000, stratified over protocol
               split) + 6 country classes x 5000 (from the just-rebuilt pool
               above; AU/IN/CN are short of 5000 so are topped up by
               server-side GCS copy of existing clips under a `dupNNNNN_`
               prefix -- oversampling with duplication, approved for this
               task only, NOT used for the country/accent pool itself)
     - fake  = ASVspoof-2019-LA spoof (35000, stratified over
               split x system_id to keep A01-A19 attack diversity)
     - real:fake = 35000:35000 exactly.
     - real rows for country-sourced clips reference the audio already
       uploaded to curated/<CC>/audio/ (`audio_uri`) instead of duplicating
       ~30k files; ASVspoof-derived real/fake rows are re-encoded to 16 kHz
       mono and copied into curated_spoof/real_fake_5k/audio_asv/.
     - train/val/test splitting is deferred to a later task -- this is a
       flat pool. `orig_split`/`system_id` provenance columns are kept so a
       later split step can still respect ASVspoof's unseen-attack boundary
       if desired.
"""
from __future__ import annotations

import concurrent.futures as cf
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict

import numpy as np
import soundfile as sf
import librosa
from google.cloud import storage

BUCKET = sys.argv[1]
LOG_PREFIX = sys.argv[2]  # "reports/dataset_rebuild_v2/<ts>" (no gs://, no bucket)
# Where to READ the pre-rebuild country clips from. The orchestrator archives
# curated/ then DELETES it before this VM starts (so the destination prefix
# "curated/" is empty when we begin) -- source must be the archive copy, not
# "curated/" itself. Defaults to "curated" only for standalone/manual reruns
# against a bucket that still has a live curated/ pool.
SOURCE_COUNTRY_PREFIX = sys.argv[3] if len(sys.argv) > 3 else "curated"

SEED = 42
SR = 16000
COUNTRY_CLASSES = ["US", "UK", "CA", "AU", "IN", "CN"]
TARGET_PER_CLASS = 5000
RF_PER_COUNTRY = 5000
RF_ASV_REAL = 5000
RF_ASV_FAKE = 35000

SEG_MIN_S = 0.4
SEG_MAX_S = 7.5
SEG_TARGET_S = 3.5
SILENCE_TOP_DB = 30
MERGE_GAP_S = 0.3
# I/O-bound (network + short-lived ffmpeg subprocesses), not CPU-bound, so
# worker counts run well above vCPU count. Country classes run CONCURRENTLY
# with each other and with the ASVspoof stage (see main()) -- each gets its
# own pool so none of them stall waiting for the others.
MAX_WORKERS = 20        # per country class
ASV_WORKERS = 48        # dedicated pool for the (largest) ASVspoof stage

client = storage.Client()
bucket = client.bucket(BUCKET)

WORK = tempfile.mkdtemp(prefix="rebuild_")


def _rng_for(tag: str) -> random.Random:
    """Fresh, deterministic RNG per call site -- avoids sharing mutable
    random.Random state across the threads now running classes concurrently."""
    return random.Random(f"{SEED}:{tag}")


def log(*a):
    print(*a, flush=True)


def download(blob_name: str, local_path: str):
    bucket.blob(blob_name).download_to_filename(local_path)


def ffmpeg_resample(src: str, dst: str):
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", src, "-ar", str(SR), "-ac", "1", dst],
        check=True,
    )


# --------------------------------------------------------------------------- #
# SAA long-paragraph -> GLOBE-like segments
# --------------------------------------------------------------------------- #
def split_saa(wav_path: str, out_dir: str, base_name: str) -> list[tuple[str, float]]:
    y, sr = sf.read(wav_path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    try:
        intervals = librosa.effects.split(y, top_db=SILENCE_TOP_DB,
                                           frame_length=2048, hop_length=512)
    except Exception:
        intervals = np.array([[0, len(y)]])
    if len(intervals) == 0:
        intervals = np.array([[0, len(y)]])

    gap = int(MERGE_GAP_S * sr)
    merged = []
    for s, e in intervals:
        if merged and s - merged[-1][1] < gap:
            merged[-1][1] = e
        else:
            merged.append([int(s), int(e)])

    segs = []
    for s, e in merged:
        dur = (e - s) / sr
        if dur <= SEG_MAX_S:
            segs.append((s, e))
        else:
            n = max(1, round(dur / SEG_TARGET_S))
            step = (e - s) // n
            for k in range(n):
                a = s + k * step
                b = e if k == n - 1 else s + (k + 1) * step
                segs.append((a, b))

    out = []
    for i, (s, e) in enumerate(segs):
        dur = (e - s) / sr
        if dur < SEG_MIN_S:
            continue
        seg = y[s:e]
        fn = f"{base_name}_seg{i:02d}.wav"
        sf.write(os.path.join(out_dir, fn), seg, sr, subtype="PCM_16")
        out.append((fn, dur))
    return out


# --------------------------------------------------------------------------- #
# country classes
# --------------------------------------------------------------------------- #
def process_country_class(cc: str) -> tuple[list[dict], str]:
    man_local = os.path.join(WORK, f"{cc}_manifest.csv")
    download(f"{SOURCE_COUNTRY_PREFIX}/{cc}/manifest.csv", man_local)
    rows = list(csv.DictReader(open(man_local, encoding="utf-8")))
    out_dir = os.path.join(WORK, "out", cc, "audio")
    os.makedirs(out_dir, exist_ok=True)
    log(f"[{cc}] {len(rows)} source clips, downloading + resampling ...")

    def handle(r: dict) -> list[dict]:
        fname = r["fname"]
        blob_name = f"{SOURCE_COUNTRY_PREFIX}/{cc}/audio/{fname}"
        tmp_in = os.path.join(WORK, f"dl_{cc}_{fname}")
        try:
            download(blob_name, tmp_in)
        except Exception as exc:
            log(f"  ! {cc}/{fname}: download failed: {exc}")
            return []
        base = os.path.splitext(fname)[0]
        tmp_wav = os.path.join(WORK, f"rs_{cc}_{base}.wav")
        try:
            ffmpeg_resample(tmp_in, tmp_wav)
        except Exception as exc:
            log(f"  ! {cc}/{fname}: resample failed: {exc}")
            return []
        finally:
            try:
                os.remove(tmp_in)
            except OSError:
                pass

        results = []
        if r.get("source") == "SAA":
            for fn, dur in split_saa(tmp_wav, out_dir, base):
                nr = dict(r)
                nr["fname"] = fn
                nr["orig_fname"] = fname
                nr["duration_s"] = round(dur, 3)
                results.append(nr)
            try:
                os.remove(tmp_wav)
            except OSError:
                pass
        else:
            final = os.path.join(out_dir, base + ".wav")
            shutil.move(tmp_wav, final)
            dur = sf.info(final).frames / SR
            nr = dict(r)
            nr["fname"] = base + ".wav"
            nr["orig_fname"] = fname
            nr["duration_s"] = round(dur, 3)
            results.append(nr)
        return results

    out_rows: list[dict] = []
    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for res in ex.map(handle, rows):
            out_rows.extend(res)
    log(f"[{cc}] {len(out_rows)} clips after resample+split "
        f"({sum(1 for r in out_rows if r['source']=='SAA')} from SAA segmentation)")
    return out_rows, out_dir


def rebalance(cc: str, rows: list[dict]) -> list[dict]:
    if len(rows) <= TARGET_PER_CLASS:
        return rows
    r_rng = _rng_for(f"rebalance:{cc}")
    by_spk = defaultdict(list)
    for r in rows:
        by_spk[r["speaker"]].append(r)
    capped = []
    for spk, rs in by_spk.items():
        saa_rs = [r for r in rs if r["source"] == "SAA"]
        other_rs = [r for r in rs if r["source"] != "SAA"]
        if len(saa_rs) > 3:
            r_rng.shuffle(saa_rs)
            saa_rs = saa_rs[:3]
        capped.extend(other_rs + saa_rs)
    r_rng.shuffle(capped)
    kept = capped[:TARGET_PER_CLASS]
    log(f"[{cc}] downsampled {len(rows)} -> {len(kept)}")
    return kept


def upload_country_class(cc: str, rows: list[dict], out_dir: str):
    fields = ["fname", "source", "speaker", "gender", "age", "accent",
              "orig_fname", "duration_s"]
    man_path = os.path.join(WORK, f"{cc}_new_manifest.csv")
    with open(man_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    bucket.blob(f"curated/{cc}/manifest.csv").upload_from_filename(man_path)

    keep = [r["fname"] for r in rows]

    def up(fn):
        local = os.path.join(out_dir, fn)
        if os.path.exists(local):
            bucket.blob(f"curated/{cc}/audio/{fn}").upload_from_filename(local)

    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        list(ex.map(up, keep))
    log(f"[{cc}] uploaded {len(keep)} audio files + manifest")


# --------------------------------------------------------------------------- #
# ASVspoof selection (manifest-only, download only the sampled subset)
# --------------------------------------------------------------------------- #
def stratified_sample(rows: list[dict], key_fn, n: int, tag: str) -> list[dict]:
    s_rng = _rng_for(f"sample:{tag}")
    groups = defaultdict(list)
    for r in rows:
        groups[key_fn(r)].append(r)
    total = len(rows)
    out = []
    for rs in groups.values():
        s_rng.shuffle(rs)
        take = round(n * len(rs) / total) if total else 0
        out.extend(rs[:take])
    s_rng.shuffle(out)
    if len(out) < n:
        chosen_ids = {id(r) for r in out}
        pool = [r for r in rows if id(r) not in chosen_ids]
        s_rng.shuffle(pool)
        out.extend(pool[: n - len(out)])
    return out[:n]


def load_asvspoof_manifests() -> tuple[list[dict], list[dict]]:
    all_rows = []
    for split in ("train", "dev", "eval"):
        local = os.path.join(WORK, f"asv_{split}_manifest.csv")
        download(f"curated_spoof/asvspoof2019_la/{split}/manifest.csv", local)
        rs = list(csv.DictReader(open(local, encoding="utf-8")))
        for r in rs:
            r["orig_split"] = split
        all_rows.extend(rs)
    bonafide = [r for r in all_rows if r["key"] == "bonafide"]
    spoof = [r for r in all_rows if r["key"] == "spoof"]
    log(f"ASVspoof source: {len(bonafide)} bonafide, {len(spoof)} spoof")
    return bonafide, spoof


def fetch_asvspoof_audio(rows: list[dict], out_dir: str) -> list[dict]:
    """Download + resample the selected clips into out_dir; returns rows that succeeded."""
    os.makedirs(out_dir, exist_ok=True)
    ok = []

    def handle(r):
        fname = r["fname"]
        blob_name = f"curated_spoof/asvspoof2019_la/{r['orig_split']}/audio/{fname}"
        tmp_in = os.path.join(WORK, f"asv_dl_{fname}")
        try:
            download(blob_name, tmp_in)
        except Exception as exc:
            log(f"  ! asv/{fname}: download failed: {exc}")
            return None
        final = os.path.join(out_dir, fname)
        try:
            ffmpeg_resample(tmp_in, final)
        except Exception as exc:
            log(f"  ! asv/{fname}: resample failed: {exc}")
            return None
        finally:
            try:
                os.remove(tmp_in)
            except OSError:
                pass
        return r

    with cf.ThreadPoolExecutor(max_workers=ASV_WORKERS) as ex:
        for res in ex.map(handle, rows):
            if res is not None:
                ok.append(res)
    return ok


# --------------------------------------------------------------------------- #
# real/fake pool assembly
# --------------------------------------------------------------------------- #
def build_real_fake_pool(country_final: dict[str, list[dict]],
                          asv_real_rows: list[dict], asv_fake_rows: list[dict]):
    # runs single-threaded after all class/ASVspoof stages finish -- safe to
    # use one RNG instance for the whole function.
    rng = _rng_for("real_fake_pool")
    rf_rows = []

    for r in asv_real_rows:
        rf_rows.append(dict(
            label="real", country="NA", source="ASVspoof", system_id="-",
            speaker=r["speaker"], orig_split=r["orig_split"], fname=r["fname"],
            audio_uri=f"gs://{BUCKET}/curated_spoof/real_fake_5k/audio_asv/{r['fname']}",
        ))
    for r in asv_fake_rows:
        rf_rows.append(dict(
            label="fake", country="NA", source="ASVspoof", system_id=r["system_id"],
            speaker=r["speaker"], orig_split=r["orig_split"], fname=r["fname"],
            audio_uri=f"gs://{BUCKET}/curated_spoof/real_fake_5k/audio_asv/{r['fname']}",
        ))

    for cc, rows in country_final.items():
        rows = rows[:]
        rng.shuffle(rows)
        if len(rows) >= RF_PER_COUNTRY:
            chosen = rows[:RF_PER_COUNTRY]
            for r in chosen:
                rf_rows.append(dict(
                    label="real", country=cc, source=r["source"], system_id="-",
                    speaker=r["speaker"], orig_split="-", fname=r["fname"],
                    audio_uri=f"gs://{BUCKET}/curated/{cc}/audio/{r['fname']}",
                ))
            continue

        for r in rows:
            rf_rows.append(dict(
                label="real", country=cc, source=r["source"], system_id="-",
                speaker=r["speaker"], orig_split="-", fname=r["fname"],
                audio_uri=f"gs://{BUCKET}/curated/{cc}/audio/{r['fname']}",
            ))
        shortfall = RF_PER_COUNTRY - len(rows)
        log(f"[{cc}] real/fake pool: only {len(rows)} unique clips, "
            f"oversampling {shortfall} via server-side duplicate copy")
        dup_pairs = []  # (src_row, dup_fname)
        i = 0
        while len(dup_pairs) < shortfall:
            r = rows[i % len(rows)]
            dup_fn = f"dup{len(dup_pairs):05d}_{r['fname']}"
            dup_pairs.append((r, dup_fn))
            i += 1

        def do_copy(pair):
            r, dup_fn = pair
            src_blob = bucket.blob(f"curated/{cc}/audio/{r['fname']}")
            bucket.copy_blob(src_blob, bucket,
                              new_name=f"curated_spoof/real_fake_5k/audio_dup/{dup_fn}")
            return r, dup_fn

        with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            for r, dup_fn in ex.map(do_copy, dup_pairs):
                rf_rows.append(dict(
                    label="real", country=cc, source=r["source"], system_id="-",
                    speaker=r["speaker"], orig_split="-", fname=dup_fn,
                    audio_uri=f"gs://{BUCKET}/curated_spoof/real_fake_5k/audio_dup/{dup_fn}",
                ))
    return rf_rows


def upload_real_fake_manifest(rf_rows: list[dict]):
    fields = ["label", "country", "source", "system_id", "speaker", "orig_split",
              "fname", "audio_uri"]
    man_path = os.path.join(WORK, "real_fake_5k_manifest.csv")
    with open(man_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rf_rows:
            w.writerow({k: r.get(k, "") for k in fields})
    bucket.blob("curated_spoof/real_fake_5k/manifest.csv").upload_from_filename(man_path)
    log(f"real/fake manifest uploaded: {len(rf_rows)} rows")


def process_and_upload_country(cc: str) -> tuple[str, list[dict]]:
    rows, out_dir = process_country_class(cc)
    rows = rebalance(cc, rows)
    upload_country_class(cc, rows, out_dir)
    shutil.rmtree(out_dir, ignore_errors=True)
    return cc, rows


def asvspoof_stage() -> tuple[list[dict], list[dict]]:
    bonafide, spoof = load_asvspoof_manifests()
    real_sel = stratified_sample(bonafide, lambda r: r["orig_split"], RF_ASV_REAL, "asv_real")
    fake_sel = stratified_sample(spoof, lambda r: (r["orig_split"], r["system_id"]),
                                  RF_ASV_FAKE, "asv_fake")
    log(f"ASVspoof selected {len(real_sel)} bonafide, {len(fake_sel)} spoof; "
        f"downloading + resampling (pool={ASV_WORKERS}) ...")
    asv_out = os.path.join(WORK, "out", "asv_audio")
    # one combined fetch call so both share the full ASV_WORKERS pool at once
    # instead of running back-to-back at the same concurrency.
    combined_ok = fetch_asvspoof_audio(real_sel + fake_sel, asv_out)
    ok_fnames = {r["fname"] for r in combined_ok}
    real_ok = [r for r in real_sel if r["fname"] in ok_fnames]
    fake_ok = [r for r in fake_sel if r["fname"] in ok_fnames]
    log(f"ASVspoof resampled ok: {len(real_ok)} bonafide, {len(fake_ok)} spoof")

    def up(fn):
        bucket.blob(f"curated_spoof/real_fake_5k/audio_asv/{fn}").upload_from_filename(
            os.path.join(asv_out, fn))

    with cf.ThreadPoolExecutor(max_workers=ASV_WORKERS) as ex:
        list(ex.map(up, ok_fnames))
    log("ASVspoof audio uploaded")
    return real_ok, fake_ok


# --------------------------------------------------------------------------- #
def main():
    log("=== running 6 country classes + ASVspoof stage CONCURRENTLY ===")
    country_final: dict[str, list[dict]] = {}
    real_ok: list[dict] = []
    fake_ok: list[dict] = []

    with cf.ThreadPoolExecutor(max_workers=len(COUNTRY_CLASSES) + 1) as top_ex:
        futures = {top_ex.submit(process_and_upload_country, cc): cc for cc in COUNTRY_CLASSES}
        asv_future = top_ex.submit(asvspoof_stage)
        futures[asv_future] = "ASVSPOOF"

        for fut in cf.as_completed(futures):
            tag = futures[fut]
            if tag == "ASVSPOOF":
                real_ok, fake_ok = fut.result()
            else:
                cc, rows = fut.result()
                country_final[cc] = rows

    log("=== assemble real/fake pool (5000/country + 5000 ASV real vs 35000 ASV fake) ===")
    rf_rows = build_real_fake_pool(country_final, real_ok, fake_ok)
    upload_real_fake_manifest(rf_rows)

    real_n = sum(1 for r in rf_rows if r["label"] == "real")
    fake_n = sum(1 for r in rf_rows if r["label"] == "fake")

    report = {
        "country_pool": {cc: len(rows) for cc, rows in country_final.items()},
        "real_fake_pool": {
            "real_total": real_n, "fake_total": fake_n,
            "real_by_country": {
                cc: sum(1 for r in rf_rows if r["label"] == "real" and r["country"] == cc)
                for cc in COUNTRY_CLASSES
            },
            "real_asv": sum(1 for r in rf_rows if r["label"] == "real" and r["country"] == "NA"),
            "fake_asv": fake_n,
        },
    }
    bucket.blob(f"{LOG_PREFIX}/rebuild_report.json").upload_from_string(
        json.dumps(report, indent=2))
    bucket.blob(f"{LOG_PREFIX}/_DONE").upload_from_string("ok\n")
    log("DONE:", json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
