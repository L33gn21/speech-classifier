"""Phase 1 (multi-task) — build unified country + real/fake split manifests.

Reads the flat, pre-balanced real/fake pool built by the v2 dataset rebuild
(``classifier/gcloud/rebuild_dataset_v2_vm.py``, DATASET.md §11):

    curated_spoof/real_fake_5k/manifest.csv
    columns: label(real/fake), country(US/UK/CA/AU/IN/CN/NA), source
             (GLOBE/SAA/ASVspoof), system_id(A01-A19 or "-"), speaker,
             orig_split(ASVspoof protocol split or "-"), fname, audio_uri

``audio_uri`` is already a full ``gs://`` path — real country-sourced rows
point straight at ``curated/<CC>/audio/``, ASVspoof-derived and oversample-dup
rows point at ``curated_spoof/real_fake_5k/audio_asv|audio_dup/`` — so this
module (and ``dataset.MultiTaskDataset``) never needs a separate root per row.

The pool is already balanced (real:fake = 35000:35000, §11) AND already has a
precomputed, speaker-disjoint ``split`` column (train/val/test, ~70:15:15 --
written by ``gcloud/pad_and_split_v2.py``: 6 country-real buckets split
independently, plus ONE joint split decision per ASVspoof speaker id shared
across bonafide/spoof so real-NA and fake-NA stay disjoint together, not just
each independently balanced). This module just reads that column.

Unified manifest columns:
    filename, audio_uri, country, country_label, fake_label, speaker,
    source, system_id, orig_split

Label assignment per row:
  - country real (GLOBE/SAA): country_label=0..5, fake_label=0 (real)
  - ASVspoof bonafide/spoof (country="NA"): country_label=-100 (ignored by
    the country loss), fake_label=0 (real) / 1 (fake)

NOTE: the precomputed split does NOT preserve ASVspoof's unseen-attack
(A07-A19 in eval) protocol boundary — whether that boundary should be
respected instead is an open decision (DATASET.md §11 "Deferred").
``system_id``/``orig_split`` are kept in the output so a future re-split can
still make that call without re-touching the bucket.
"""
# 1단계(멀티태스크) — 국가 + real/fake 통합 스플릿 매니페스트 생성.
# v2 데이터셋 재구축(rebuild_dataset_v2_vm.py, DATASET.md §11)이 만든 평탄한
# real/fake 풀(curated_spoof/real_fake_5k/manifest.csv)을 읽는다. 이 풀은
# 이미 real:fake=35000:35000 로 균형이 맞춰져 있고, 화자 단위로 미리 계산된
# split 컬럼(train/val/test, 약 70:15:15 — gcloud/pad_and_split_v2.py)도 이미
# 갖고 있다. 여기서 분할을 다시 계산하지 않고 그 컬럼을 그대로 읽는다.
# audio_uri 가 이미 완전한 gs:// 경로이므로 root+subdir 조합이 필요 없다.
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import (
    COUNTRY_IGNORE_INDEX,
    FAKE2ID,
    LABEL2ID,
    REAL_FAKE_ROOT,
    SEED,
    TEST_FRACTION,
    VAL_FRACTION,
)

UNIFIED_COLS = ["filename", "audio_uri", "country", "country_label", "fake_label",
                "speaker", "source", "system_id", "orig_split", "split"]


def load_real_fake_pool(real_fake_root: Path) -> pd.DataFrame:
    """Read curated_spoof/real_fake_5k/manifest.csv (raw source schema)."""
    mpath = Path(real_fake_root) / "manifest.csv"
    df = pd.read_csv(mpath, dtype=str, keep_default_na=False)
    if "split" not in df.columns:
        raise ValueError(
            f"{mpath} has no 'split' column — run gcloud/pad_and_split_v2.py "
            "first (DATASET.md §11)")
    return df


def _to_unified(df: pd.DataFrame) -> pd.DataFrame:
    """Map the raw real_fake_5k manifest -> the unified training schema."""
    # country="NA" (ASVspoof-sourced rows) -> no country label (ignored by the
    # country loss); the 6 country codes map to their usual 0..5 ids.
    country_label = df["country"].map(lambda c: LABEL2ID.get(c, COUNTRY_IGNORE_INDEX))
    fake_label = df["label"].map(FAKE2ID)
    return pd.DataFrame({
        "filename": df["fname"].to_numpy(),
        "audio_uri": df["audio_uri"].to_numpy(),
        "country": df["country"].to_numpy(),
        "country_label": country_label.astype(int).to_numpy(),
        "fake_label": fake_label.astype(int).to_numpy(),
        "speaker": df["speaker"].to_numpy(),
        "source": df["source"].to_numpy(),
        "system_id": df["system_id"].to_numpy() if "system_id" in df else "-",
        "orig_split": df["orig_split"].to_numpy() if "orig_split" in df else "-",
        "split": df["split"].to_numpy(),
    })


def build_multitask_splits(
    real_fake_root: Path = REAL_FAKE_ROOT,
    val_fraction: float = VAL_FRACTION,
    test_fraction: float = TEST_FRACTION,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the pre-balanced real/fake pool, split by its precomputed ``split``
    column. ``val_fraction``/``test_fraction``/``seed`` are accepted for call-site
    compatibility but unused -- the split is already fixed in the manifest."""
    raw = load_real_fake_pool(real_fake_root)
    df = _to_unified(raw)
    train = df[df["split"] == "train"].reset_index(drop=True)
    val = df[df["split"] == "val"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)

    tr, va, te = set(train["speaker"]), set(val["speaker"]), set(test["speaker"])
    assert not (tr & va), f"speaker leakage train/val: {len(tr & va)}"
    assert not (tr & te), f"speaker leakage train/test: {len(tr & te)}"
    assert not (va & te), f"speaker leakage val/test: {len(va & te)}"
    return train[UNIFIED_COLS], val[UNIFIED_COLS], test[UNIFIED_COLS]


def report(name: str, df: pd.DataFrame) -> None:
    """Print the country and real/fake composition of a unified split."""
    real = int((df["fake_label"] == FAKE2ID["real"]).sum())
    fake = int((df["fake_label"] == FAKE2ID["fake"]).sum())
    has_country = int((df["country_label"] != COUNTRY_IGNORE_INDEX).sum())
    print(f"[{name}] {len(df)} clips  country-labeled={has_country}  "
          f"real={real} fake={fake}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    ap.add_argument("--test-fraction", type=float, default=TEST_FRACTION)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--real-fake-root", default=str(REAL_FAKE_ROOT))
    ap.add_argument("--manifest-dir", default=None)
    args = ap.parse_args()

    train, val, test = build_multitask_splits(
        args.real_fake_root, args.val_fraction, args.test_fraction, args.seed)
    for nm, part in [("train", train), ("val", val), ("test", test)]:
        report(nm, part)

    if args.manifest_dir:
        out = Path(args.manifest_dir)
        out.mkdir(parents=True, exist_ok=True)
        for nm, part in [("train", train), ("val", val), ("test", test)]:
            dest = out / f"{nm}.csv"
            part.to_csv(dest, index=False)
            print(f"wrote {dest} ({len(part)} rows)")


if __name__ == "__main__":
    main()
