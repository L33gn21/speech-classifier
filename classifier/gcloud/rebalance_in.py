"""Downsample IN's Svarah clips (session-level) so curated/IN matches US/UK scale.

Strategy: keep every non-Svarah IN row untouched. For Svarah, downsample by
whole recording *session* (the `speaker` column is a session id, not a
person - see DATASET.md §5.3), stratified per language (`accent` column,
e.g. Svarah-Hindi) so all 19 languages shrink by roughly the same fraction
instead of some languages disappearing. Deterministic: seed=42, matching the
project's existing convention (curate.py notes in DATASET.md).
"""
import pandas as pd
import numpy as np

SEED = 42
TARGET_TOTAL = 5500  # ~matches US 5081 / UK 5812

df = pd.read_csv("IN_manifest.csv", dtype=str)
svarah = df[df["source"] == "Svarah"].copy()
non_svarah = df[df["source"] != "Svarah"].copy()

target_svarah = TARGET_TOTAL - len(non_svarah)
frac = target_svarah / len(svarah)
print(f"non-Svarah rows: {len(non_svarah)}")
print(f"Svarah rows: {len(svarah)} -> target {target_svarah} ({frac:.1%})")

rng = np.random.default_rng(SEED)
kept_parts = []
report_rows = []
for accent, group in svarah.groupby("accent"):
    sessions = group["speaker"].unique()
    rng.shuffle(sessions)
    lang_target_clips = round(len(group) * frac)
    acc = 0
    chosen_sessions = []
    for s in sessions:
        if acc >= lang_target_clips:
            break
        chosen_sessions.append(s)
        acc += (group["speaker"] == s).sum()
    kept = group[group["speaker"].isin(chosen_sessions)]
    kept_parts.append(kept)
    report_rows.append((accent, len(group), len(kept), group["speaker"].nunique(), len(chosen_sessions)))

svarah_kept = pd.concat(kept_parts)
out = pd.concat([non_svarah, svarah_kept]).reset_index(drop=True)

print(f"\n{'accent':22s} {'before':>7s} {'after':>7s} {'sess_before':>12s} {'sess_after':>11s}")
for r in sorted(report_rows, key=lambda x: -x[1]):
    print(f"{r[0]:22s} {r[1]:7d} {r[2]:7d} {r[3]:12d} {r[4]:11d}")

print(f"\nTotal IN before: {len(df)}  after: {len(out)}")
print(f"Svarah before: {len(svarah)}  after: {len(svarah_kept)}")

out.to_csv("IN_manifest_rebalanced.csv", index=False)
print("\nwrote IN_manifest_rebalanced.csv")
