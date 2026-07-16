#!/usr/bin/env python
"""Build curated/<CC>/ classes from the raw GLOBE + SAA pool (us-west2 bucket).

Reads the raw archive laid down by the ingest VM:

    gs://<bucket>/raw/globe/data/*.parquet         (audio + accent,speaker_id,gender,age,transcript)
    gs://<bucket>/raw/saa/audio/saa_*.mp3          (1 clip per speaker)
    gs://<bucket>/raw/saa/saa_manifest.csv         (fname,source,speaker,gender,age,accent,native_language,country,birthplace)

and writes, per class code <CC> chosen in the spec:

    <out>/<CC>/manifest.csv   columns: fname,source,speaker,gender,age,accent   (DATASET.md schema)
    <out>/<CC>/audio/<fname>

Country label is the FOLDER, never a column (matches the training loader in
src/dataset.py + src/prepare_data.py). Source is preserved in the manifest and
as the fname prefix (glb_ / saa_) for channel-leakage analysis.

Curation recipe (same shape as the original project, DATASET.md §4):
  - GLOBE  (the volume backbone): up to N_G female + N_G male speakers
           (default 100+100), <= CAP_G clips/speaker (default 30), gender balanced.
  - SAA    (speaker diversity):   up to N_S female + N_S male speakers
           (default 40+40), 1 clip each (SAA is 1 clip/speaker already).
  - Deterministic (seed). Speakers namespaced by source. fname collisions across
    ALL classes are checked and rejected.

The class -> source mapping is entirely spec-driven (--spec spec.json) so no code
change is needed once the class list + label axis are decided:

    {
      "axis": "country",                 // "country" | "language"  (SAA filter column)
      "seed": 42,
      "globe_speakers_per_gender": 100,
      "globe_clips_per_speaker": 30,
      "saa_speakers_per_gender": 40,
      "classes": {
        "NZ": {"globe_accents": ["New Zealand English"],
               "saa_countries": ["new zealand"], "saa_languages": []},
        "PH": {"globe_accents": ["Filipino"],
               "saa_countries": ["philippines"], "saa_languages": ["tagalog"]},
        "DE": {"globe_accents": ["German English,Non native speaker"],
               "saa_countries": ["germany"], "saa_languages": ["german"]}
      }
    }

GLOBE `accent` strings must match EXACTLY (they are messy Common-Voice labels —
see reports/globe_report.json for the exact values + counts). SAA countries /
languages match the lower-cased value of the chosen axis column.

Runs on a GCE VM in us-west2 (same region as the bucket -> no egress). Example:
    python3 curate.py --bucket qi-ucsd-speech-usw2 --spec spec.json --out /mnt/work/curated
    gcloud storage cp -r /mnt/work/curated/* gs://qi-ucsd-speech-usw2/curated/
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
from collections import defaultdict

import pyarrow.parquet as pq


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def log(*a):
    print(*a, flush=True)


def sniff_ext(b: bytes) -> str:
    """Guess an audio container from magic bytes (GLOBE clips are re-encoded)."""
    if b[:4] == b"RIFF":
        return "wav"
    if b[:4] == b"fLaC":
        return "flac"
    if b[:4] == b"OggS":
        return "ogg"
    if b[:3] == b"ID3" or b[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "mp3"
    return "wav"


def norm_gender(v: str) -> str:
    return {"male": "M", "female": "F", "m": "M", "f": "F"}.get((v or "").strip().lower(), "U")


def gsutil_ls(uri: str) -> list[str]:
    out = subprocess.run(["gcloud", "storage", "ls", uri], capture_output=True, text=True)
    if out.returncode != 0:
        return []
    return [l.strip() for l in out.stdout.splitlines() if l.strip()]


def gsutil_cp(src: str, dst: str):
    subprocess.run(["gcloud", "storage", "cp", src, dst], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# --------------------------------------------------------------------------- #
# GLOBE — two-pass over the parquet shards
#   pass 1: scan (accent, speaker_id, gender) only -> decide kept (shard,row) set
#   pass 2: re-scan, extract audio bytes for kept rows, write files
# --------------------------------------------------------------------------- #
def globe_select(work_globe: list[str], spec: dict, rng: random.Random):
    """Return {cc: {(shard_idx,row_idx): (speaker,gender,age)}} of clips to keep."""
    accent_to_cc = {}
    for cc, c in spec["classes"].items():
        for a in c.get("globe_accents", []):
            accent_to_cc[a] = cc
    if not accent_to_cc:
        return {cc: {} for cc in spec["classes"]}, {}

    cap_spk = spec.get("globe_speakers_per_gender", 100)
    cap_clip = spec.get("globe_clips_per_speaker", 30)

    # pass 1 — gather candidate clips per class, grouped by (speaker, gender)
    # cand[cc][speaker] = {"gender":g, "clips":[(shard,row,age), ...]}
    cand = {cc: defaultdict(lambda: {"gender": "U", "clips": []}) for cc in spec["classes"]}
    for si, local in enumerate(work_globe):
        pf = pq.ParquetFile(local)
        names = set(pf.schema_arrow.names)
        cols = [c for c in ("accent", "speaker_id", "gender", "age") if c in names]
        row = 0
        for b in pf.iter_batches(batch_size=16384, columns=cols):
            d = b.to_pydict()
            accs = d.get("accent")
            n = len(accs) if accs is not None else 0
            spks = d.get("speaker_id"); gens = d.get("gender"); ages = d.get("age")
            for i in range(n):
                cc = accent_to_cc.get(accs[i])
                if cc is not None:
                    spk = f"glb_{spks[i]}" if spks is not None else f"glb_s{si}_{row}"
                    g = norm_gender(gens[i]) if gens is not None else "U"
                    e = cand[cc][spk]
                    e["gender"] = g
                    e["clips"].append((si, row, (ages[i] if ages is not None else "")))
                row += 1
        log(f"  globe pass1 shard {si+1}/{len(work_globe)} scanned")

    # decide kept rows: gender-balanced speaker cap, then per-speaker clip cap
    kept = {cc: {} for cc in spec["classes"]}
    summary = {}
    for cc in spec["classes"]:
        by_gender = {"F": [], "M": [], "U": []}
        for spk, e in cand[cc].items():
            by_gender.setdefault(e["gender"], []).append(spk)
        chosen = []
        for g in ("F", "M"):
            spks = by_gender.get(g, [])
            rng.shuffle(spks)
            chosen += spks[:cap_spk]
        # if a gender is thin, backfill from "U" so we don't waste the budget
        budget = 2 * cap_spk - len(chosen)
        if budget > 0:
            u = by_gender.get("U", []); rng.shuffle(u); chosen += u[:budget]
        nclip = 0
        for spk in chosen:
            e = cand[cc][spk]
            clips = e["clips"][:]
            rng.shuffle(clips)
            clips = clips[:cap_clip]
            for (si, row, age) in clips:
                kept[cc][(si, row)] = (spk, e["gender"], str(age))
            nclip += len(clips)
        summary[cc] = {"globe_speakers": len(chosen), "globe_clips": nclip}
        log(f"  GLOBE {cc}: {len(chosen)} speakers, {nclip} clips")
    return kept, summary


def globe_extract(work_globe: list[str], kept: dict, out_root: str, manifests: dict):
    """Pass 2: write audio bytes for kept (shard,row) rows + append manifest rows."""
    # invert: per shard -> {row: (cc, speaker, gender, age)}
    per_shard = defaultdict(dict)
    for cc, rows in kept.items():
        for (si, row), (spk, g, age) in rows.items():
            per_shard[si][row] = (cc, spk, g, age)
    for si, local in enumerate(work_globe):
        want = per_shard.get(si)
        if not want:
            continue
        pf = pq.ParquetFile(local)
        names = set(pf.schema_arrow.names)
        cols = [c for c in ("audio", "accent") if c in names]
        row = 0
        for b in pf.iter_batches(batch_size=16384, columns=cols):
            d = b.to_pydict()
            audio = d.get("audio")
            accs = d.get("accent")
            n = len(audio) if audio is not None else 0
            for i in range(n):
                tgt = want.get(row)
                if tgt is not None:
                    cc, spk, g, age = tgt
                    a = audio[i]
                    blob = a["bytes"] if isinstance(a, dict) else a
                    if blob:
                        ext = sniff_ext(blob)
                        fname = f"{spk}_{si:03d}_{row}.{ext}"
                        adir = os.path.join(out_root, cc, "audio")
                        os.makedirs(adir, exist_ok=True)
                        with open(os.path.join(adir, fname), "wb") as f:
                            f.write(blob)
                        accent = accs[i] if accs is not None else "GLOBE"
                        manifests[cc].append(
                            dict(fname=fname, source="GLOBE", speaker=spk,
                                 gender=g, age=age, accent=f"GLOBE-{accent}"))
                row += 1
        log(f"  globe pass2 shard {si+1}/{len(work_globe)} extracted")


# --------------------------------------------------------------------------- #
# SAA — copy 1 clip per matching speaker, gender-balanced
# --------------------------------------------------------------------------- #
def saa_build(bucket: str, spec: dict, rng: random.Random, out_root: str, manifests: dict):
    """HYBRID SAA filter: each class picks its own axis.

    Per class, ``saa_by`` = "country" (filter native/region classes by birthplace
    country) or "language" (filter L2 classes by native_language). Defaults to
    the global spec ``axis``; if unset, inferred from which list the class fills.
    """
    default_axis = spec.get("axis", "country")
    cap_spk = spec.get("saa_speakers_per_gender", 40)
    man_uri = f"gs://{bucket}/raw/saa/saa_manifest.csv"
    local_man = "/tmp/saa_manifest.csv"
    try:
        gsutil_cp(man_uri, local_man)
    except Exception:
        log("  ! SAA manifest not found; skipping SAA")
        return {}
    rows = list(csv.DictReader(open(local_man, encoding="utf-8")))

    # A single SAA speaker can match two classes under the hybrid axis (e.g. a
    # USA-born Mandarin speaker matches US by country AND CN by language). Assign
    # each file to the FIRST class in spec order (region classes precede CN), so
    # the ambiguous clip is not double-counted / leaked across labels.
    used = {r["fname"] for rowlist in manifests.values() for r in rowlist}

    summary = {}
    for cc, c in spec["classes"].items():
        # decide this class's SAA axis
        saa_by = c.get("saa_by")
        if saa_by is None:
            has_c, has_l = bool(c.get("saa_countries")), bool(c.get("saa_languages"))
            saa_by = "language" if (has_l and not has_c) else (
                     "country" if (has_c and not has_l) else default_axis)
        if saa_by == "language":
            key, wanted = "native_language", {v.strip().lower() for v in c.get("saa_languages", [])}
        else:
            key, wanted = "country", {v.strip().lower() for v in c.get("saa_countries", [])}
        if not wanted:
            summary[cc] = {"saa_speakers": 0, "saa_clips": 0}
            continue

        cand = [r for r in rows if (r.get(key) or "").strip().lower() in wanted]
        bg = {"F": [], "M": [], "U": []}
        for r in cand:
            bg.setdefault(norm_gender(r.get("gender")), []).append(r)
        chosen = []
        for g in ("F", "M"):
            lst = bg.get(g, []); rng.shuffle(lst); chosen += lst[:cap_spk]
        budget = 2 * cap_spk - len(chosen)
        if budget > 0:
            u = bg.get("U", []); rng.shuffle(u); chosen += u[:budget]
        summary[cc] = {"saa_speakers": len(chosen), "saa_clips": len(chosen)}
        log(f"  SAA {cc} (by {saa_by}): {len(chosen)} speakers/clips")

        adir = os.path.join(out_root, cc, "audio")
        os.makedirs(adir, exist_ok=True)
        for r in chosen:
            fn = r["fname"]                       # saa_<name>.mp3
            if fn in used:                        # already claimed by an earlier class
                continue
            try:
                gsutil_cp(f"gs://{bucket}/raw/saa/audio/{fn}", os.path.join(adir, fn))
            except Exception:
                continue
            used.add(fn)
            manifests[cc].append(
                dict(fname=fn, source="SAA", speaker=r.get("speaker", ""),
                     gender=norm_gender(r.get("gender")), age=r.get("age", ""),
                     accent=r.get("accent", "SAA")))
    return summary


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--spec", required=True, help="class spec JSON")
    ap.add_argument("--out", default="/mnt/work/curated")
    ap.add_argument("--globe-dir", default="/mnt/work/globe_parquet",
                    help="local dir to stage parquet shards")
    args = ap.parse_args()

    spec = json.load(open(args.spec, encoding="utf-8"))
    rng = random.Random(spec.get("seed", 42))
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.globe_dir, exist_ok=True)
    manifests = {cc: [] for cc in spec["classes"]}

    need_globe = any(c.get("globe_accents") for c in spec["classes"].values())
    gsum = {}
    if need_globe:
        shards = gsutil_ls(f"gs://{args.bucket}/raw/globe/data/")
        shards = [s for s in shards if s.endswith(".parquet")]
        log(f"GLOBE: {len(shards)} parquet shards")
        # stage shards one at a time in each pass to bound disk (~0.5GB peak)
        # pass 1 + pass 2 both need every shard, so download once and keep? 46GB.
        # -> keep on a big disk; download all first.
        local_shards = []
        for s in shards:
            base = s.rstrip("/").split("/")[-1]
            dst = os.path.join(args.globe_dir, base)
            if not (os.path.exists(dst) and os.path.getsize(dst) > 0):
                gsutil_cp(s, dst)
            local_shards.append(dst)
            log(f"  staged {len(local_shards)}/{len(shards)} {base}")
        kept, gsum = globe_select(local_shards, spec, rng)
        globe_extract(local_shards, kept, args.out, manifests)

    ssum = saa_build(args.bucket, spec, rng, args.out, manifests)

    # safety net: guarantee each fname lives in exactly one class. Any residual
    # cross-class duplicate is dropped (kept in the first class in spec order),
    # never aborted — a lone hybrid-axis overlap must not sink a 30-min job.
    seen = set()
    for cc in spec["classes"]:
        deduped, dropped = [], 0
        for r in manifests.get(cc, []):
            if r["fname"] in seen:
                dropped += 1
                continue
            seen.add(r["fname"])
            deduped.append(r)
        if dropped:
            log(f"  {cc}: dropped {dropped} cross-class duplicate fname(s)")
        manifests[cc] = deduped

    report = {}
    for cc, rows in manifests.items():
        rows.sort(key=lambda x: x["fname"])
        cdir = os.path.join(args.out, cc)
        os.makedirs(cdir, exist_ok=True)
        with open(os.path.join(cdir, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["fname", "source", "speaker", "gender", "age", "accent"])
            w.writeheader()
            w.writerows(rows)
        nspk = len({r["speaker"] for r in rows})
        gc = {g: sum(1 for r in rows if r["gender"] == g) for g in ("F", "M", "U")}
        srcs = {}
        for r in rows:
            srcs[r["source"]] = srcs.get(r["source"], 0) + 1
        report[cc] = {"clips": len(rows), "speakers": nspk, "gender": gc, "sources": srcs}
        log(f"[{cc}] {len(rows)} clips  {nspk} spk  F/M/U={gc['F']}/{gc['M']}/{gc['U']}  {srcs}")

    with open(os.path.join(args.out, "curation_report.json"), "w", encoding="utf-8") as f:
        json.dump({"spec": spec, "classes": report}, f, indent=2, ensure_ascii=False)
    log("DONE. curated at", args.out)


if __name__ == "__main__":
    main()
