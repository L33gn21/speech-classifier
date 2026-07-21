"""Phase 1 (multi-task) — build unified country + real/fake split manifests.

Combines the validated country splits (``prepare_data.build_splits`` over the
GLOBE/SAA curated pool) with the ASVspoof 2019 LA spoof corpus (``curated_spoof``)
into one train/val/test manifest for joint training of the country head + the
real/fake head.

Unified manifest columns:
    filename, dataset, subdir, country_label, fake_label, speaker, source

Label assignment per clip:
  - accent (GLOBE/SAA): dataset=accent, subdir=<CC>,    country_label=0..5, fake_label=0 (real)
  - ASVspoof bonafide : dataset=spoof,  subdir=<split>, country_label=-100, fake_label=0 (real)
  - ASVspoof spoof    : dataset=spoof,  subdir=<split>, country_label=-100, fake_label=1 (fake)

Split mapping preserves the ASVspoof protocol (eval = unseen attacks A07-A19):
    accent train + asvspoof train -> train
    accent val   + asvspoof dev   -> val
    accent test  + asvspoof eval  -> test   (held-out; eval attacks unseen in train)

The spoof corpus keeps ALL bonafide (the channel-matched real anchor) and caps
only the much larger spoof set per split via ``spoof_cap``. Random clip sampling
within a split is leakage-safe (ASVspoof splits are already speaker-disjoint by
protocol) and preserves attack-system coverage.
"""
# 1단계(멀티태스크) — 국가 + real/fake 통합 스플릿 매니페스트 생성.
# 검증된 국가 스플릿(prepare_data.build_splits, GLOBE/SAA)에 ASVspoof 2019 LA spoof
# 코퍼스(curated_spoof)를 합쳐 국가 헤드 + real/fake 헤드를 함께 학습할 train/val/test
# 를 만든다. ASVspoof 프로토콜 스플릿(eval=미지 공격)을 보존한다.
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    COUNTRY_IGNORE_INDEX,
    CURATED_ROOT,
    FAKE2ID,
    SEED,
    SPOOF_ROOT,
    TARGET_PER_CLASS,
    TEST_FRACTION,
    VAL_FRACTION,
)
from prepare_data import build_splits

UNIFIED_COLS = ["filename", "dataset", "subdir", "country_label",
                "fake_label", "speaker", "source"]

# our split name -> ASVspoof protocol split folder under curated_spoof/
# 우리 스플릿 이름 -> curated_spoof 아래 ASVspoof 프로토콜 스플릿 폴더명.
_SPOOF_SPLIT_FOR = {"train": "train", "val": "dev", "test": "eval"}


def _accent_to_unified(df: pd.DataFrame) -> pd.DataFrame:
    """Map a country split frame (filename,label,country,speaker,source) -> unified."""
    # 국가 스플릿 프레임을 통합 스키마로 변환한다(전부 real, 국가 라벨 0..5).
    return pd.DataFrame({
        "filename": df["filename"].to_numpy(),
        "dataset": "accent",
        "subdir": df["country"].to_numpy(),
        "country_label": df["label"].astype(int).to_numpy(),
        "fake_label": FAKE2ID["real"],
        "speaker": df["speaker"].to_numpy(),
        "source": df["source"].to_numpy() if "source" in df else "",
    })


def _load_spoof_split(spoof_root: Path, split: str, spoof_cap: int | None,
                      rng: np.random.Generator) -> pd.DataFrame:
    """Read curated_spoof/<split>/manifest.csv -> unified (bonafide=real, spoof=fake).

    Keeps ALL bonafide (channel-matched real anchor); caps only spoof to
    ``spoof_cap`` via random within-split sampling (leakage-safe, see module doc).
    """
    # curated_spoof/<split>/manifest.csv (컬럼: fname,source,speaker,key,system_id,split)
    # 를 읽어 통합 스키마로. key: bonafide->real(0), spoof->fake(1). bonafide 는 전량
    # 유지(채널 매칭 real 앵커), spoof 만 spoof_cap 으로 랜덤 언더샘플(스플릿 내 화자
    # disjoint 이므로 클립 랜덤 샘플은 누수 없음, 공격 시스템 커버리지 보존).
    mpath = Path(spoof_root) / split / "manifest.csv"
    m = pd.read_csv(mpath, dtype=str, keep_default_na=False)
    is_bona = m["key"].str.lower() == "bonafide"
    m = m.assign(_fake=np.where(is_bona, FAKE2ID["real"], FAKE2ID["fake"]))
    bona = m[m["_fake"] == FAKE2ID["real"]]
    spoof = m[m["_fake"] == FAKE2ID["fake"]]
    if spoof_cap is not None and len(spoof) > spoof_cap:
        idx = rng.choice(spoof.index.to_numpy(), size=int(spoof_cap), replace=False)
        spoof = spoof.loc[idx]
    keep = pd.concat([bona, spoof])
    return pd.DataFrame({
        "filename": keep["fname"].to_numpy(),
        "dataset": "spoof",
        "subdir": split,
        "country_label": COUNTRY_IGNORE_INDEX,
        "fake_label": keep["_fake"].astype(int).to_numpy(),
        "speaker": keep["speaker"].to_numpy(),
        "source": (keep["source"].to_numpy() if "source" in keep else "ASVspoof2019LA"),
    })


def build_multitask_splits(
    curated_root: Path = CURATED_ROOT,
    spoof_root: Path = SPOOF_ROOT,
    per_class: int = TARGET_PER_CLASS,
    val_fraction: float = VAL_FRACTION,
    test_fraction: float = TEST_FRACTION,
    spoof_cap: int | None = None,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Country splits (validated recipe) + ASVspoof protocol splits -> unified 3-way."""
    # 국가 스플릿(검증 레시피) + ASVspoof 프로토콜 스플릿을 통합해 3분할 반환.
    rng = np.random.default_rng(seed)
    a_train, a_val, a_test = build_splits(
        curated_root, per_class, val_fraction, test_fraction, seed)
    out = {}
    for name, adf in [("train", a_train), ("val", a_val), ("test", a_test)]:
        acc = _accent_to_unified(adf)
        spf = _load_spoof_split(spoof_root, _SPOOF_SPLIT_FOR[name], spoof_cap, rng)
        out[name] = pd.concat([acc, spf])[UNIFIED_COLS].reset_index(drop=True)
    return out["train"], out["val"], out["test"]


def report(name: str, df: pd.DataFrame) -> None:
    """Print the accent/spoof and real/fake composition of a unified split."""
    # 통합 스플릿의 accent/spoof, real/fake 구성을 출력한다(진단용).
    acc = int((df["dataset"] == "accent").sum())
    spf = int((df["dataset"] == "spoof").sum())
    real = int((df["fake_label"] == FAKE2ID["real"]).sum())
    fake = int((df["fake_label"] == FAKE2ID["fake"]).sum())
    print(f"[{name}] {len(df)} clips  accent={acc} spoof={spf}  real={real} fake={fake}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=TARGET_PER_CLASS)
    ap.add_argument("--spoof-cap", "--spoof_cap", dest="spoof_cap", type=int, default=None,
                    help="cap spoof clips per split (bonafide always kept in full)")
    ap.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    ap.add_argument("--test-fraction", type=float, default=TEST_FRACTION)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--curated-root", default=str(CURATED_ROOT))
    ap.add_argument("--spoof-root", default=str(SPOOF_ROOT))
    ap.add_argument("--manifest-dir", default=None)
    args = ap.parse_args()

    train, val, test = build_multitask_splits(
        args.curated_root, args.spoof_root, args.per_class,
        args.val_fraction, args.test_fraction, args.spoof_cap, args.seed)
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
