"""Phase 1 — dataset collection from the curated GCS pool.

Build train/val/test manifests directly from the per-country
``curated/<CC>/manifest.csv`` files described in DATASET.md §11.

Each class is already fixed at exactly 5000 clips (AU/IN/CN padded up by
duplicate-copy) with a precomputed, speaker-disjoint ``split`` column
(train/val/test, ~70:15:15 -- see ``gcloud/pad_and_split_v2.py``). This module
just reads that column; it does NOT recompute the split. `--per-class` still
lets a run cap clips for quick experiments (speaker-aware, applied within
each split so the 70:15:15 shape is preserved).

Output columns (train/val/test): filename,label,country,speaker,source

The curated pool itself is never modified — we only read its manifests and
write split manifests under MANIFEST_DIR. Runs the same locally or against a
FUSE-mounted GCS bucket on Vertex AI (see config.gcs_to_fuse).
"""
# 1단계 — curated GCS 풀로부터 데이터셋 수집/전처리.
#
# DATASET.md §11 에 기술된 국가별 curated/<CC>/manifest.csv 파일들을 읽는다.
# 각 클래스는 이미 정확히 5000개로 고정돼 있고(AU/IN/CN 은 중복 복사로 패딩),
# 화자 단위로 미리 계산된 split 컬럼(train/val/test, 약 70:15:15 —
# gcloud/pad_and_split_v2.py 참고)을 그대로 읽기만 한다. 여기서 분할을 다시
# 계산하지 않는다. --per-class 는 빠른 실험용으로 각 분할 내에서 화자 단위로
# 클립 수를 제한하는 용도로만 남아 있다(70:15:15 비율은 유지됨).
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
    TEST_FRACTION,
    VAL_FRACTION,
)


def load_curated(curated_root: Path = CURATED_ROOT) -> pd.DataFrame:
    """Read every curated/<CC>/manifest.csv into one labelled dataframe.

    Returns columns: filename, country, label, speaker, source, split.
    """
    # 각 국가 폴더의 manifest.csv 를 읽어 하나의 라벨링된 데이터프레임으로 합친다.
    frames = []
    for cc in LABELS:
        mpath = Path(curated_root) / cc / "manifest.csv"
        if not mpath.exists():
            print(f"  ! {cc}: manifest not found at {mpath} — skipping")
            continue
        m = pd.read_csv(mpath, dtype=str, keep_default_na=False)
        if "split" not in m.columns:
            raise ValueError(
                f"{mpath} has no 'split' column — run gcloud/pad_and_split_v2.py "
                "first (DATASET.md §11)")
        # speaker 컬럼이 비어 있는 행은 파일명을 화자 id 대용으로 사용(화자 단위 분할 안정성).
        speaker = m["speaker"].where(m["speaker"].astype(bool), other=cc + "_" + m["fname"])
        part = pd.DataFrame(
            {
                "filename": m["fname"],
                "country": cc,
                "label": LABEL2ID[cc],
                "speaker": speaker,
                "source": m.get("source", pd.Series([""] * len(m))),
                "split": m["split"],
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


def build_splits(
    curated_root: Path = CURATED_ROOT,
    per_class: int | None = None,
    val_fraction: float = VAL_FRACTION,
    test_fraction: float = TEST_FRACTION,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load curated pool, split by the precomputed ``split`` column.

    ``per_class`` (optional) caps clips for a quick experiment, applied
    speaker-aware WITHIN each split so the fixed 70:15:15 shape carries over
    (target ~per_class*0.70/0.15/0.15 per split). Omit it to use every clip.
    """
    # curated 로드 -> 미리 계산된 split 컬럼으로 3분할. per_class 는 각 분할 내부에서
    # 화자 단위로 클립 수를 줄이는 선택적 스모크테스트용 캡일 뿐, 분할 자체는 고정값을 그대로 쓴다.
    df = load_curated(curated_root)
    print(f"loaded {len(df)} clips across {df['country'].nunique()} classes")
    train = df[df["split"] == "train"].reset_index(drop=True)
    val = df[df["split"] == "val"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)

    if per_class:
        rng = np.random.default_rng(seed)
        train = balanced_sample(train, int(round(per_class * (1 - val_fraction - test_fraction))), rng)
        val = balanced_sample(val, int(round(per_class * val_fraction)), rng)
        test = balanced_sample(test, int(round(per_class * test_fraction)), rng)
        print(f"capped to ~{per_class}/class: train={len(train)} val={len(val)} test={len(test)}")

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
    ap.add_argument("--per-class", type=int, default=None,
                    help="optional speaker-aware cap for quick experiments; "
                         "omit to use the full fixed 5000/class pool (DATASET.md §11)")
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
