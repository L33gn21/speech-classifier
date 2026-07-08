"""Central configuration for the Level 1 accent classifier.

All paths are resolved relative to the project root so the scripts work
regardless of the current working directory.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "cv-corpus-26.0-2026-06-12" / "en"
CLIPS_DIR = DATA_ROOT / "clips"
VALIDATED_TSV = DATA_ROOT / "validated.tsv"

MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# ---------------------------------------------------------------------------
# Label space
# ---------------------------------------------------------------------------
# The `accents` column holds free-form descriptive strings, not short codes.
# We map the exact single-accent strings to our 4 target classes. Rows whose
# `accents` value is a mix (contains "|") are dropped to avoid ambiguity.
ACCENT_MAP: dict[str, str] = {
    "United States English": "us",
    "England English": "england",
    "India and South Asia (India, Pakistan, Sri Lanka)": "indian",
    "Australian English": "australia",
}

# Fixed order -> integer label id. Keep stable; the trained head depends on it.
LABELS: list[str] = ["us", "england", "indian", "australia"]
LABEL2ID: dict[str, int] = {name: i for i, name in enumerate(LABELS)}
ID2LABEL: dict[int, str] = {i: name for name, i in LABEL2ID.items()}
NUM_LABELS = len(LABELS)

# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16_000
MAX_DURATION_S = 8.0  # crop cap; collator pads to per-batch max up to this
MAX_SAMPLES = int(SAMPLE_RATE * MAX_DURATION_S)

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
MIN_UP_VOTES = 2
MAX_DOWN_VOTES = 0
TARGET_PER_CLASS = 4_000          # balanced under-sampling target
TEST_FRACTION = 0.15              # speaker-level holdout fraction
SEED = 42

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_NAME = "facebook/wav2vec2-base"
