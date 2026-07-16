"""Phase 1 — dataset collection from the curated GCS pool.

Build balanced, speaker-disjoint train/val/test manifests directly from the
per-country ``curated/<CC>/manifest.csv`` files described in DATASET.md.

Recipe:
  1. read each curated/<CC>/manifest.csv (columns: fname,source,speaker,gender,age,accent)
  2. label = the folder name <CC> (US/UK/IN/NG/CA/JP/CN)
  3. balanced under-sampling: at most TARGET_PER_CLASS clips per class
     (small classes like CA/JP keep all their clips), speaker-aware so a
     capped class keeps whole speakers rather than slicing one speaker in half
  4. speaker-level split into train/val/test (no speaker in two splits)

Output columns (train/val/test): filename,label,country,speaker,source

The curated pool itself is never modified — we only read its manifests and
write split manifests under MANIFEST_DIR. Runs the same locally or against a
FUSE-mounted GCS bucket on Vertex AI (see config.gcs_to_fuse).
"""
# 1단계 — curated GCS 풀로부터 데이터셋 수집/전처리.
#
# DATASET.md 에 기술된 국가별 curated/<CC>/manifest.csv 파일들로부터
# 클래스 균형이 맞고 화자(speaker)가 겹치지 않는 train/val/test 매니페스트를 만든다.
#
# 처리 순서:
#   1. 각 curated/<CC>/manifest.csv 읽기 (컬럼: fname,source,speaker,gender,age,accent)
#   2. 라벨 = 폴더 이름 <CC> (US/UK/IN/NG/CA/JP/CN)
#   3. 클래스 균형 언더샘플링: 클래스당 최대 TARGET_PER_CLASS 개
#      (CA/JP 처럼 작은 클래스는 전량 유지). 화자 단위로 뽑아 한 화자가 잘리지 않게 함.
#   4. 화자 단위로 train/val/test 분할 (같은 화자가 두 분할에 동시에 등장하지 않음)
#
# 원본 curated 풀은 절대 수정하지 않는다 — 매니페스트를 읽기만 하고,
# 분할 결과만 MANIFEST_DIR 아래에 기록한다.
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    CURATED_ROOT,
    LABEL2ID,
    LABELS,
    MANIFEST_DIR,
    MAX_CLIPS_PER_SPEAKER,
    SEED,
    TARGET_PER_CLASS,
    TEST_FRACTION,
    VAL_FRACTION,
)


def load_curated(curated_root: Path = CURATED_ROOT) -> pd.DataFrame:
    """Read every curated/<CC>/manifest.csv into one labelled dataframe.

    Returns columns: filename, country, label, speaker, source.
    """
    # 각 국가 폴더의 manifest.csv 를 읽어 하나의 라벨링된 데이터프레임으로 합친다.
    frames = []
    for cc in LABELS:
        mpath = Path(curated_root) / cc / "manifest.csv"
        if not mpath.exists():
            print(f"  ! {cc}: manifest not found at {mpath} — skipping")
            continue
        m = pd.read_csv(mpath, dtype=str, keep_default_na=False)
        # speaker 컬럼이 비어 있는 행은 파일명을 화자 id 대용으로 사용(화자 단위 분할 안정성).
        speaker = m["speaker"].where(m["speaker"].astype(bool), other=cc + "_" + m["fname"])
        part = pd.DataFrame(
            {
                "filename": m["fname"],
                "country": cc,
                "label": LABEL2ID[cc],
                "speaker": speaker,
                "source": m.get("source", pd.Series([""] * len(m))),
            }
        )
        frames.append(part)
        print(f"  {cc}: {len(part)} clips, {part['speaker'].nunique()} speakers")
    if not frames:
        raise FileNotFoundError(f"no manifests found under {curated_root}")
    return pd.concat(frames).reset_index(drop=True)


def cap_per_speaker(
    sub: pd.DataFrame, max_per_speaker: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Keep at most ``max_per_speaker`` random clips for each speaker."""
    # 화자별로 최대 max_per_speaker 개의 클립만 무작위로 남긴다. 한 화자가 수백 개
    # 클립을 가진 소스(SpeechOcean762 등)가 클래스/분할을 지배하는 것을 막는다.
    if max_per_speaker <= 0 or sub.empty:
        return sub
    parts = []
    for _, group in sub.groupby("speaker"):
        if len(group) > max_per_speaker:
            idx = rng.choice(group.index.to_numpy(), size=max_per_speaker, replace=False)
            parts.append(group.loc[idx])
        else:
            parts.append(group)
    return pd.concat(parts)


def balanced_sample(
    df: pd.DataFrame,
    per_class: int,
    rng: np.random.Generator,
    max_per_speaker: int = MAX_CLIPS_PER_SPEAKER,
) -> pd.DataFrame:
    """Cap clips per speaker, then cap each class at ``per_class`` clips.

    Speakers are added whole so a downstream speaker-disjoint split stays clean.
    Small classes keep all their clips.
    """
    # 1) 화자당 클립 상한 적용 -> 2) 클래스당 클립 상한 적용(화자를 통째로 추가).
    # 화자 단위로 추가하므로 뒤의 화자 단위 분할이 깔끔하게 된다. 작은 클래스는 전량 유지.
    parts = []
    for cc in LABELS:
        sub = df[df["country"] == cc]
        sub = cap_per_speaker(sub, max_per_speaker, rng)
        if len(sub) <= per_class:
            parts.append(sub)
            continue
        # sizes as a plain dict: pandas Series label-indexing with a numpy-scalar
        # key (from .index.to_numpy()) can silently return the wrong value, which
        # made the accumulator undercount and let a class blow past the cap.
        sizes = sub.groupby("speaker").size().to_dict()
        speakers = list(sizes.keys())
        rng.shuffle(speakers)
        kept: list[str] = []
        acc = 0
        for s in speakers:
            if acc >= per_class:
                break
            kept.append(s)
            acc += sizes[s]
        parts.append(sub[sub["speaker"].isin(kept)])
    return pd.concat(parts).reset_index(drop=True)


def speaker_split(
    df: pd.DataFrame,
    val_fraction: float,
    test_fraction: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Assign whole speakers to val/test per class (speaker-disjoint splits)."""
    # 클래스별로 화자를 통째로 val/test 에 배정한다. 같은 화자의 목소리가
    # 여러 분할에 동시에 나타나 모델이 억양이 아니라 목소리를 외우는
    # 데이터 누수(leakage)를 막기 위함이다 (DATASET.md §5.2).
    val_sp: set[str] = set()
    test_sp: set[str] = set()
    for cc in LABELS:
        sub = df[df["country"] == cc]
        # plain dict (not pandas scalar-label indexing) — see balanced_sample note.
        sizes = sub.groupby("speaker").size().to_dict()
        speakers = list(sizes.keys())
        rng.shuffle(speakers)
        n = len(sub)
        val_target = int(round(n * val_fraction))
        test_target = int(round(n * test_fraction))
        acc = 0
        i = 0
        # 먼저 test 목표치를 채우고, 이어서 val 목표치를 채운다. 나머지는 train.
        while i < len(speakers) and acc < test_target:
            test_sp.add(speakers[i])
            acc += sizes[speakers[i]]
            i += 1
        acc = 0
        while i < len(speakers) and acc < val_target:
            val_sp.add(speakers[i])
            acc += sizes[speakers[i]]
            i += 1
    is_test = df["speaker"].isin(test_sp)
    is_val = df["speaker"].isin(val_sp)
    train = df[~is_test & ~is_val].reset_index(drop=True)
    val = df[is_val].reset_index(drop=True)
    test = df[is_test].reset_index(drop=True)
    return train, val, test


def build_splits(
    curated_root: Path = CURATED_ROOT,
    per_class: int = TARGET_PER_CLASS,
    val_fraction: float = VAL_FRACTION,
    test_fraction: float = TEST_FRACTION,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """End-to-end: load curated pool -> balance -> speaker-disjoint 3-way split."""
    # curated 로드 -> 클래스 균형 언더샘플링 -> 화자 단위 3분할 까지 한 번에 수행.
    rng = np.random.default_rng(seed)
    df = load_curated(curated_root)
    print(f"loaded {len(df)} clips across {df['country'].nunique()} classes")
    df = balanced_sample(df, per_class, rng)
    print(f"after balanced under-sampling (cap {per_class}/class): {len(df)} clips")
    train, val, test = speaker_split(df, val_fraction, test_fraction, rng)

    # sanity: no speaker in more than one split (no leakage)
    # 안전장치: 어떤 화자도 두 개 이상의 분할에 동시에 존재하지 않는지 최종 검증.
    tr, va, te = set(train["speaker"]), set(val["speaker"]), set(test["speaker"])
    assert not (tr & va), f"speaker leakage train/val: {len(tr & va)}"
    assert not (tr & te), f"speaker leakage train/test: {len(tr & te)}"
    assert not (va & te), f"speaker leakage val/test: {len(va & te)}"
    return train, val, test


def report(name: str, df: pd.DataFrame) -> None:
    # 분할 결과(클립 수, 화자 수)를 클래스별로 콘솔에 출력하는 진단용 함수.
    counts = df["country"].value_counts().reindex(LABELS, fill_value=0)
    speakers = df.groupby("country")["speaker"].nunique().reindex(LABELS, fill_value=0)
    print(f"[{name}] {len(df)} clips")
    for cc in LABELS:
        print(f"    {cc:4s} clips={counts[cc]:6d}  speakers={speakers[cc]:5d}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=TARGET_PER_CLASS)
    ap.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    ap.add_argument("--test-fraction", type=float, default=TEST_FRACTION)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--manifest-dir", default=str(MANIFEST_DIR))
    args = ap.parse_args()

    train, val, test = build_splits(
        CURATED_ROOT, args.per_class, args.val_fraction, args.test_fraction, args.seed
    )
    report("train", train)
    report("val", val)
    report("test", test)

    out_dir = Path(args.manifest_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["filename", "label", "country", "speaker", "source"]
    for name, part in [("train", train), ("val", val), ("test", test)]:
        dest = out_dir / f"{name}.csv"
        part[cols].to_csv(dest, index=False)
        print(f"wrote {dest} ({len(part)} rows)")


if __name__ == "__main__":
    main()
