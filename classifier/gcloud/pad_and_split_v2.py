"""Runs LOCALLY (metadata-only + server-side GCS copies, no bulk audio transit).

Two jobs, both operating purely on manifests + server-side `copy_blob` calls
(bytes never leave GCS, so this is safe to run off the project machine per
CLAUDE.md §2 -- nothing here is a bulk audio download):

1. Pad AU/IN/CN in `curated/<CC>/` up to exactly 5000 clips each (they were
   left short after the 2026-07-22 rebuild -- 4984/4560/1689) by duplicating
   existing clips (`dupNNNNN_<fname>`), cycling through the available pool.
   US/UK/CA are already exactly 5000, untouched.
2. Add a speaker-disjoint `split` column (train/val/test, ~70:15:15) to both
   `curated/<CC>/manifest.csv` (stratified per class) and
   `curated_spoof/real_fake_5k/manifest.csv` (stratified per bucket: 6
   country-real buckets + one combined ASVspoof bucket -- bonafide/spoof
   share the same speaker-id namespace in ASVspoof's protocol, so real-NA and
   fake-NA get ONE joint per-speaker split decision to stay disjoint across
   the whole pool, not just within each bucket).
"""
from __future__ import annotations

import concurrent.futures as cf
import csv
import io
import random
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict

# On Windows, `gcloud` is a .cmd shim -- subprocess needs the resolved path
# (or shell=True) to find it; plain "gcloud" fails CreateProcess directly.
GCLOUD = shutil.which("gcloud") or "gcloud"

# Uses the `gcloud storage` CLI (already authenticated in this shell) instead
# of the google-cloud-storage python client, which needs separate Application
# Default Credentials that aren't set up on this machine. `gcloud storage cp`
# between two gs:// paths is a server-side copy -- no bulk audio transits this
# machine either way (CLAUDE.md §2).

BUCKET = "qi-ucsd-speech-usw2"
SEED = 42
COUNTRY_CLASSES = ["US", "UK", "CA", "AU", "IN", "CN"]
TARGET_PER_CLASS = 5000
SPLIT_RATIO = {"train": 0.70, "val": 0.15, "test": 0.15}


def log(*a):
    print(*a, flush=True)


def gs(path: str) -> str:
    return f"gs://{BUCKET}/{path}"


def gcp_copy(src: str, dst: str):
    last_err = None
    for attempt in range(4):
        r = subprocess.run([GCLOUD, "storage", "cp", gs(src), gs(dst)],
                            capture_output=True, text=True)
        if r.returncode == 0:
            return
        last_err = r.stderr
        time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"gcp_copy failed after retries: {src} -> {dst}\n{last_err}")


def read_manifest(blob_name: str) -> list[dict]:
    out = subprocess.run([GCLOUD, "storage", "cat", gs(blob_name)],
                          check=True, capture_output=True, text=True, encoding="utf-8")
    return list(csv.DictReader(io.StringIO(out.stdout)))


def write_manifest(blob_name: str, rows: list[dict], fields: list[str]):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in fields})
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False,
                                      encoding="utf-8", newline="") as f:
        f.write(buf.getvalue())
        tmp_path = f.name
    subprocess.run([GCLOUD, "storage", "cp", tmp_path, gs(blob_name)],
                    check=True, capture_output=True)


def greedy_speaker_split(weights: dict[str, int], seed_tag: str) -> dict[str, str]:
    """weights: speaker_id -> clip count. Returns speaker_id -> split, balanced
    to ~70:15:15 by largest-remaining-deficit greedy assignment."""
    rng = random.Random(f"{SEED}:{seed_tag}")
    items = list(weights.items())
    rng.shuffle(items)
    total = sum(w for _, w in items)
    targets = {s: total * r for s, r in SPLIT_RATIO.items()}
    counts = {s: 0.0 for s in SPLIT_RATIO}
    assign = {}
    for spk, w in items:
        best = max(counts, key=lambda s: targets[s] - counts[s])
        assign[spk] = best
        counts[best] += w
    return assign


# --------------------------------------------------------------------------- #
# 1) pad AU/IN/CN to 5000 (server-side duplicate copy)
# --------------------------------------------------------------------------- #
def pad_class(cc: str, rows: list[dict]) -> list[dict]:
    if len(rows) >= TARGET_PER_CLASS:
        return rows
    shortfall = TARGET_PER_CLASS - len(rows)
    log(f"[{cc}] padding {len(rows)} -> {TARGET_PER_CLASS} (+{shortfall}, duplicate copy)")
    base_rows = rows[:]
    plan = []
    for k in range(shortfall):
        r = base_rows[k % len(base_rows)]
        dup_fn = f"dup{k:05d}_{r['fname']}"
        plan.append((r, dup_fn))

    def do_copy(pair):
        r, dup_fn = pair
        gcp_copy(f"curated/{cc}/audio/{r['fname']}", f"curated/{cc}/audio/{dup_fn}")
        nr = dict(r)
        nr["fname"] = dup_fn
        nr["orig_fname"] = r.get("orig_fname") or r["fname"]
        return nr

    padded = []
    with cf.ThreadPoolExecutor(max_workers=10) as ex:
        futures = [ex.submit(do_copy, pair) for pair in plan]
        for fut in cf.as_completed(futures):
            padded.append(fut.result())
    return rows + padded


# --------------------------------------------------------------------------- #
# 2a) country pool split (per-class, speaker-disjoint)
# --------------------------------------------------------------------------- #
def split_country_class(cc: str, rows: list[dict]) -> list[dict]:
    weights = defaultdict(int)
    for r in rows:
        weights[r["speaker"]] += 1
    assign = greedy_speaker_split(dict(weights), seed_tag=f"country:{cc}")
    for r in rows:
        r["split"] = assign[r["speaker"]]
    counts = defaultdict(int)
    for r in rows:
        counts[r["split"]] += 1
    log(f"[{cc}] split: train={counts['train']} val={counts['val']} test={counts['test']} "
        f"({len(weights)} speakers)")
    return rows


# --------------------------------------------------------------------------- #
# 2b) real/fake pool split
# --------------------------------------------------------------------------- #
def split_real_fake(rows: list[dict]) -> list[dict]:
    country_rows = [r for r in rows if r["country"] != "NA"]
    asv_rows = [r for r in rows if r["country"] == "NA"]

    for cc in COUNTRY_CLASSES:
        sub = [r for r in country_rows if r["country"] == cc]
        weights = defaultdict(int)
        for r in sub:
            weights[r["speaker"]] += 1
        assign = greedy_speaker_split(dict(weights), seed_tag=f"rf_country:{cc}")
        for r in sub:
            r["split"] = assign[r["speaker"]]
        counts = defaultdict(int)
        for r in sub:
            counts[r["split"]] += 1
        log(f"[real_fake/{cc}] split: train={counts['train']} val={counts['val']} "
            f"test={counts['test']} ({len(weights)} speakers)")

    # ASVspoof bonafide + spoof share the speaker-id namespace (a "speaker" is
    # a target identity with both genuine and spoofed utterances) -- one joint
    # per-speaker assignment so real-NA and fake-NA stay disjoint together,
    # not just independently balanced.
    weights = defaultdict(int)
    for r in asv_rows:
        weights[r["speaker"]] += 1
    assign = greedy_speaker_split(dict(weights), seed_tag="rf_asv_joint")
    for r in asv_rows:
        r["split"] = assign[r["speaker"]]
    counts = defaultdict(lambda: defaultdict(int))
    for r in asv_rows:
        counts[r["label"]][r["split"]] += 1
    for label in ("real", "fake"):
        c = counts[label]
        log(f"[real_fake/ASV-{label}] split: train={c['train']} val={c['val']} "
            f"test={c['test']} ({len(weights)} shared speakers)")

    return country_rows + asv_rows


# --------------------------------------------------------------------------- #
def main():
    log("=== 1) pad AU/IN/CN to 5000, then speaker-disjoint 70:15:15 split ===")
    country_fields = ["fname", "source", "speaker", "gender", "age", "accent",
                       "orig_fname", "duration_s", "split"]
    for cc in COUNTRY_CLASSES:
        rows = read_manifest(f"curated/{cc}/manifest.csv")
        rows = pad_class(cc, rows)
        rows = split_country_class(cc, rows)
        rows.sort(key=lambda r: r["fname"])
        write_manifest(f"curated/{cc}/manifest.csv", rows, country_fields)
        log(f"[{cc}] final: {len(rows)} clips, manifest updated")

    log("=== 2) real/fake pool: speaker-disjoint 70:15:15 split ===")
    rf_fields = ["label", "country", "source", "system_id", "speaker",
                 "orig_split", "fname", "audio_uri", "split"]
    rf_rows = read_manifest("curated_spoof/real_fake_5k/manifest.csv")
    rf_rows = split_real_fake(rf_rows)
    rf_rows.sort(key=lambda r: (r["label"], r["country"], r["fname"]))
    write_manifest("curated_spoof/real_fake_5k/manifest.csv", rf_rows, rf_fields)
    log(f"real/fake pool: {len(rf_rows)} rows, manifest updated")

    log("DONE.")


if __name__ == "__main__":
    main()
