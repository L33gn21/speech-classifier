"""Wave 0 — build honest multitask eval manifests (metadata-only, run once).

Produces two unified test.csv files (schema: filename, audio_uri, country,
country_label, fake_label, speaker, source, system_id, orig_split, split)
that evaluate.py's multitask path (evaluate_multitask) can score directly:

  0a. unseen-attack fake bench — ASVspoof 2019 LA `eval` split (A07-A19,
      speaker/attack-disjoint from train/dev). Source of truth for "did fake
      detection actually get better", since real_fake_5k's own test split does
      NOT preserve this boundary (DATASET.md §11).
  0b. VoxForge false-fake bench — all-real cross-domain speech, to measure the
      real head's false-fake rate (does it wrongly flag non-ASVspoof real
      speech as synthetic).

Reads only small manifest CSVs (already downloaded via `gcloud storage cat`
by the caller shell script) — no bulk audio ever transits this machine
(CLAUDE.md §2). Writes local CSVs; the caller uploads them with
`gcloud storage cp`.

Usage:
    python build_eval_manifests_wave0.py \
        --asvspoof-eval-manifest <local asvspoof eval manifest.csv> \
        --voxforge-manifest <local voxforge test.csv> \
        --out-dir <local output dir>
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

ASV_AUDIO_PREFIX = "gs://qi-ucsd-speech-usw2/curated_spoof/asvspoof2019_la/eval/audio"
VOXFORGE_AUDIO_ROOT = "gs://qi-ucsd-speech-usc1/test_voxforge"
LABELS = ["US", "UK", "CA", "AU", "IN", "CN"]
COUNTRY_IGNORE_INDEX = -100

UNIFIED_COLS = [
    "filename", "audio_uri", "country", "country_label", "fake_label",
    "speaker", "source", "system_id", "orig_split", "split",
]


def build_unseen_attack_bench(asvspoof_eval_manifest: str) -> pd.DataFrame:
    df = pd.read_csv(asvspoof_eval_manifest)
    assert set(df["split"].unique()) == {"eval"}, "expected only the ASVspoof 'eval' split"
    out = pd.DataFrame({
        "filename": df["fname"],
        "audio_uri": ASV_AUDIO_PREFIX + "/" + df["fname"],
        "country": "NA",
        "country_label": COUNTRY_IGNORE_INDEX,
        "fake_label": (df["key"] == "spoof").astype(int),
        "speaker": df["speaker"],
        "source": df["source"],
        "system_id": df["system_id"],
        "orig_split": df["split"],
        "split": "test",
    })
    return out[UNIFIED_COLS]


def build_voxforge_false_fake_bench(voxforge_manifest: str) -> pd.DataFrame:
    df = pd.read_csv(voxforge_manifest)
    out = pd.DataFrame({
        "filename": df["filename"],
        "audio_uri": VOXFORGE_AUDIO_ROOT + "/" + df["country"] + "/audio/" + df["filename"],
        "country": df["country"],
        "country_label": df["label"],
        "fake_label": 0,
        "speaker": df["speaker"],
        "source": df["source"],
        "system_id": "-",
        "orig_split": "voxforge",
        "split": "test",
    })
    return out[UNIFIED_COLS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asvspoof-eval-manifest", required=True)
    ap.add_argument("--voxforge-manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out_dir, "unseen_attack_fake"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "voxforge_false_fake"), exist_ok=True)

    unseen = build_unseen_attack_bench(args.asvspoof_eval_manifest)
    unseen_path = os.path.join(args.out_dir, "unseen_attack_fake", "test.csv")
    unseen.to_csv(unseen_path, index=False)
    print(f"unseen-attack bench: {len(unseen)} rows -> {unseen_path}")
    print(f"  fake_label counts:\n{unseen['fake_label'].value_counts()}")

    vox = build_voxforge_false_fake_bench(args.voxforge_manifest)
    vox_path = os.path.join(args.out_dir, "voxforge_false_fake", "test.csv")
    vox.to_csv(vox_path, index=False)
    print(f"\nvoxforge false-fake bench: {len(vox)} rows -> {vox_path}")
    print(f"  country_label counts:\n{vox['country_label'].value_counts()}")


if __name__ == "__main__":
    main()
