"""Build balanced, speaker-disjoint train/test manifests from validated.tsv.

Recipe (see CLAUDE.md):
  1. load validated.tsv
  2. keep rows whose `accents` maps to one of the 4 target classes
  3. quality filter: up_votes >= 2 and down_votes == 0
  4. balanced under-sampling: TARGET_PER_CLASS clips per accent
  5. speaker-level split by client_id (no speaker in both train and test)

Output: data/manifests/{train,test}.csv with columns
        filename,label,accent,client_id
"""
from __future__ import annotations

import argparse
import csv

import numpy as np
import pandas as pd

from config import (
    ACCENT_MAP,
    CLIPS_DIR,
    LABEL2ID,
    LABELS,
    MANIFEST_DIR,
    MAX_DOWN_VOTES,
    MIN_UP_VOTES,
    SEED,
    TARGET_PER_CLASS,
    TEST_FRACTION,
    VALIDATED_TSV,
)


def load_validated() -> pd.DataFrame:
    cols = ["client_id", "path", "up_votes", "down_votes", "accents"]
    df = pd.read_csv(
        VALIDATED_TSV,
        sep="\t",
        usecols=cols,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        quoting=csv.QUOTE_NONE,
    )
    df["up_votes"] = pd.to_numeric(df["up_votes"], errors="coerce").fillna(0).astype(int)
    df["down_votes"] = pd.to_numeric(df["down_votes"], errors="coerce").fillna(0).astype(int)
    return df


def filter_and_label(df: pd.DataFrame) -> pd.DataFrame:
    # exact single-accent match only (drop mixed "a|b" rows implicitly)
    df = df[df["accents"].isin(ACCENT_MAP)].copy()
    df["accent"] = df["accents"].map(ACCENT_MAP)
    df = df[(df["up_votes"] >= MIN_UP_VOTES) & (df["down_votes"] <= MAX_DOWN_VOTES)]
    return df


def balanced_sample(df: pd.DataFrame, per_class: int, rng: np.random.Generator) -> pd.DataFrame:
    parts = []
    for accent in LABELS:
        sub = df[df["accent"] == accent]
        n = min(per_class, len(sub))
        if n < per_class:
            print(f"  ! {accent}: only {len(sub)} clips available (< {per_class})")
        idx = rng.choice(sub.index.to_numpy(), size=n, replace=False)
        parts.append(sub.loc[idx])
    return pd.concat(parts).reset_index(drop=True)


def speaker_split(
    df: pd.DataFrame, test_fraction: float, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign whole speakers to test until each class hits ~test_fraction clips."""
    test_clients: set[str] = set()
    for accent in LABELS:
        sub = df[df["accent"] == accent]
        target = int(round(len(sub) * test_fraction))
        counts = sub.groupby("client_id").size()
        clients = counts.index.to_numpy()
        rng.shuffle(clients)
        acc = 0
        for c in clients:
            if acc >= target:
                break
            test_clients.add(c)
            acc += int(counts[c])
    is_test = df["client_id"].isin(test_clients)
    return df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True)


def report(name: str, df: pd.DataFrame) -> None:
    counts = df["accent"].value_counts().reindex(LABELS, fill_value=0)
    speakers = df.groupby("accent")["client_id"].nunique().reindex(LABELS, fill_value=0)
    print(f"[{name}] {len(df)} clips")
    for a in LABELS:
        print(f"    {a:10s} clips={counts[a]:6d}  speakers={speakers[a]:5d}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=TARGET_PER_CLASS)
    ap.add_argument("--test-fraction", type=float, default=TEST_FRACTION)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--check-exists",
        action="store_true",
        help="drop rows whose mp3 is not yet extracted (slow; extraction may be ongoing)",
    )
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"Loading {VALIDATED_TSV} ...")
    df = load_validated()
    print(f"  {len(df)} validated rows")

    df = filter_and_label(df)
    print(f"  {len(df)} rows after accent + quality filter")

    if args.check_exists:
        exists = df["path"].map(lambda p: (CLIPS_DIR / p).exists())
        dropped = int((~exists).sum())
        if dropped:
            print(f"  dropping {dropped} rows with missing mp3 (not yet extracted?)")
        df = df[exists]

    df = balanced_sample(df, args.per_class, rng)
    train_df, test_df = speaker_split(df, args.test_fraction, rng)

    # sanity: no speaker leakage
    leak = set(train_df["client_id"]) & set(test_df["client_id"])
    assert not leak, f"speaker leakage: {len(leak)} clients in both splits"

    report("train", train_df)
    report("test", test_df)

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train_df), ("test", test_df)]:
        out = part[["path", "accent", "client_id"]].rename(columns={"path": "filename"})
        out["label"] = out["accent"].map(LABEL2ID)
        out = out[["filename", "label", "accent", "client_id"]]
        dest = MANIFEST_DIR / f"{name}.csv"
        out.to_csv(dest, index=False)
        print(f"wrote {dest} ({len(out)} rows)")


if __name__ == "__main__":
    main()
