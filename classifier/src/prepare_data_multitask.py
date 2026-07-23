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

NOTE: the precomputed ``split`` column is speaker-disjoint but attack-blind --
train/val/test all mix A01-A19 (DATASET.md §11), so fake metrics on that
``test`` are optimistically biased (a model can just recognize attacks it
already saw in train). ``build_multitask_splits`` fixes this in place, folded
into the same train/val/test (no 4th bucket): fake rows are re-assigned by
``system_id`` tier instead of the precomputed column --

  - FAKE_TEST_ONLY_SYSTEMS: 100% test, never train/val (genuinely unseen
    attacks at eval time).
  - FAKE_MIXED_SYSTEMS: split across train/val/test like normal (contributes
    a smaller, still partly-optimistic in-split signal for these systems).
  - everything else: train/val only, never test.

A01-A06 (orig ASVspoof train/dev, 30 speakers) and A07-A19 (orig eval, 48
speakers) use two completely disjoint speaker pools (verified against the
manifest), so as long as the test-only/mixed tiers are drawn from A07-A19,
there is zero speaker overlap with the train-only tier. Within A07-A19 itself
the same 48 speakers appear under every system, so a speaker can still show
up in both train (via a train/val-only system) and test (via a test-only
system) -- a voice-familiarity leak, strictly weaker than attack-type leakage,
same tradeoff already accepted for country-real speaker reuse across systems.
Real rows are untouched (kept on the precomputed column) since bonafide isn't
attack-bound.
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

import random

from config import (
    COUNTRY_IGNORE_INDEX,
    FAKE2ID,
    FAKE_MIXED_SYSTEMS,
    FAKE_MIXED_TEST_FRACTION,
    FAKE_TEST_ONLY_SYSTEMS,
    LABEL2ID,
    REAL_FAKE_ROOT,
    SEED,
    TEST_FRACTION,
    VAL_FRACTION,
    gcs_to_fuse,
)

UNIFIED_COLS = ["filename", "audio_uri", "country", "country_label", "fake_label",
                "speaker", "source", "system_id", "orig_split", "split"]


def load_real_fake_pool(real_fake_root: Path) -> pd.DataFrame:
    """Read curated_spoof/real_fake_5k/manifest.csv (raw source schema)."""
    mpath = Path(gcs_to_fuse(str(real_fake_root))) / "manifest.csv"
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


def _assign_by_speaker(speakers: list[str], fractions: dict[str, float],
                        seed: int) -> dict[str, str]:
    """Deterministically bucket a speaker list into fractions.keys() so every
    clip from the same speaker lands in the same bucket. fractions must sum
    to (approximately) 1."""
    ordered = sorted(speakers)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n = len(ordered)
    assignment: dict[str, str] = {}
    start = 0
    items = list(fractions.items())
    for i, (bucket, frac) in enumerate(items):
        count = n - start if i == len(items) - 1 else round(n * frac)
        for spk in ordered[start:start + count]:
            assignment[spk] = bucket
        start += count
    return assignment


def build_multitask_splits(
    real_fake_root: Path = REAL_FAKE_ROOT,
    val_fraction: float = VAL_FRACTION,
    test_fraction: float = TEST_FRACTION,
    seed: int = SEED,
    fake_test_only_systems: frozenset[str] = FAKE_TEST_ONLY_SYSTEMS,
    fake_mixed_systems: frozenset[str] = FAKE_MIXED_SYSTEMS,
    fake_mixed_test_fraction: float = FAKE_MIXED_TEST_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the pre-balanced real/fake pool. Real rows keep the precomputed
    ``split`` column (bonafide isn't attack-bound). Fake rows are instead
    re-assigned by ``system_id`` tier (see module docstring) so test.csv
    itself carries a genuinely-unseen-attack slice, without a 4th bucket:

      - fake_test_only_systems -> 100% test
      - fake_mixed_systems -> split train/val/test as usual (test share =
        fake_mixed_test_fraction, remainder split train:val at the same
        ratio as train_fraction:val_fraction)
      - everything else -> split train/val only (never test), at
        train_fraction:val_fraction

    ``test_fraction``/``val_fraction`` drive both the "everything else" and
    the "mixed" train:val split; ``fake_mixed_test_fraction`` is the extra
    knob that controls how large the honest unseen-attack slice of test is.
    """
    raw = load_real_fake_pool(real_fake_root)
    df = _to_unified(raw)

    real = df[df["fake_label"] == FAKE2ID["real"]]
    fake = df[df["fake_label"] == FAKE2ID["fake"]]

    test_only = fake[fake["system_id"].isin(fake_test_only_systems)]
    mixed = fake[fake["system_id"].isin(fake_mixed_systems)]
    rest = fake[~fake["system_id"].isin(fake_test_only_systems | fake_mixed_systems)]

    train_val_ratio = {
        "train": (1 - val_fraction - test_fraction) / (1 - test_fraction),
        "val": val_fraction / (1 - test_fraction),
    }
    rest_assign = _assign_by_speaker(list(rest["speaker"].unique()), train_val_ratio, seed)
    rest_split = rest["speaker"].map(rest_assign)

    mixed_ratio = {
        "test": fake_mixed_test_fraction,
        "train": (1 - fake_mixed_test_fraction) * train_val_ratio["train"],
        "val": (1 - fake_mixed_test_fraction) * train_val_ratio["val"],
    }
    mixed_assign = _assign_by_speaker(list(mixed["speaker"].unique()), mixed_ratio, seed + 1)
    mixed_split = mixed["speaker"].map(mixed_assign)

    fake_train = pd.concat([rest[rest_split == "train"], mixed[mixed_split == "train"]])
    fake_val = pd.concat([rest[rest_split == "val"], mixed[mixed_split == "val"]])
    fake_test = pd.concat([test_only, mixed[mixed_split == "test"]])

    train = pd.concat([real[real["split"] == "train"], fake_train]).reset_index(drop=True)
    val = pd.concat([real[real["split"] == "val"], fake_val]).reset_index(drop=True)
    test = pd.concat([real[real["split"] == "test"], fake_test]).reset_index(drop=True)

    # Real stays fully speaker-disjoint (unchanged precomputed column). Fake
    # is only disjoint ACROSS tiers by construction (test-only/mixed systems
    # are drawn from A07-A19, disjoint from the A01-A06 pool used by "rest"
    # when rest includes any A01-A06 rows) -- WITHIN A07-A19 the same 48
    # speakers appear under every system, so a speaker can legitimately land
    # in both train (via a train/val-only system) and test (via a
    # test-only/mixed system). That's an accepted, weaker voice-familiarity
    # leak (see module docstring), not asserted against here.
    real_tr, real_va, real_te = (set(real[real["split"] == s]["speaker"])
                                 for s in ("train", "val", "test"))
    assert not (real_tr & real_va), f"real speaker leakage train/val: {len(real_tr & real_va)}"
    assert not (real_tr & real_te), f"real speaker leakage train/test: {len(real_tr & real_te)}"
    assert not (real_va & real_te), f"real speaker leakage val/test: {len(real_va & real_te)}"

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
