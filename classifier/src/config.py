"""Central configuration for the accent/country classifier.

Data source is the curated GCS pool documented in DATASET.md (us-west2 rebuild):

    gs://qi-ucsd-speech-usw2/curated/<CC>/manifest.csv   (fname,source,speaker,gender,age,accent)
    gs://qi-ucsd-speech-usw2/curated/<CC>/audio/<fname>

The country label is the folder name (``<CC>``), not a column.

Paths are environment-driven so the same code runs locally and on Vertex AI:

- Locally, sensible defaults under the repo root are used.
- On Vertex AI Custom Training, buckets are FUSE-mounted at ``/gcs/<bucket>``
  and the job output dir is provided via ``AIP_MODEL_DIR`` (a ``gs://`` URI).
  Set ``CV_CURATED_ROOT`` to the ``gs://`` (or mounted ``/gcs``) curated path.

Env vars (all optional):
    CV_CURATED_ROOT   root holding <CC>/manifest.csv and <CC>/audio/
                      (default: <repo>/curated)
    CV_OUTPUT_DIR     where the trained model is written
                      (default: AIP_MODEL_DIR if set, else <repo>/outputs/classifier)
    CV_MODEL_NAME     pretrained wav2vec2 backbone (default facebook/wav2vec2-base)
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Path resolution helpers
# ---------------------------------------------------------------------------
# config.py lives at classifier/src/config.py -> repo root is three levels up.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def gcs_to_fuse(path: str) -> str:
    """Translate a ``gs://bucket/x`` URI to its Vertex AI FUSE mount ``/gcs/bucket/x``.

    Vertex AI Custom Training auto-mounts accessible buckets under ``/gcs``.
    Non-``gs://`` paths are returned unchanged.
    """
    if path.startswith("gs://"):
        return "/gcs/" + path[len("gs://"):]
    return path


def _env_path(name: str, default: Path) -> Path:
    # If the environment variable `name` is set, use its value (converted to the FUSE path
    # when it is a GCS path); otherwise use the given default path as-is.
    raw = os.environ.get(name)
    if raw:
        return Path(gcs_to_fuse(raw))
    return default


# ---------------------------------------------------------------------------
# Paths (env-overridable)
# ---------------------------------------------------------------------------
# Root directory of the curated pool; contains <CC>/manifest.csv and <CC>/audio/.
CURATED_ROOT = _env_path("CV_CURATED_ROOT", REPO_ROOT / "curated")

# Directory where the train.csv / val.csv / test.csv split manifests are stored.
# The original curated/ is never modified; only the split results are written here
# (under outputs by default).
MANIFEST_DIR = _env_path(
    "CV_MANIFEST_DIR", REPO_ROOT / "outputs" / "classifier" / "manifests"
)

# Order in which the default output directory is resolved:
# 1) the AIP_MODEL_DIR environment variable that Vertex AI passes automatically
# 2) otherwise outputs/classifier in the local repository
_default_output = os.environ.get("AIP_MODEL_DIR") or str(REPO_ROOT / "outputs" / "classifier")
# Final output directory where the trained model (weights, config, logs, ...) is saved.
OUTPUT_DIR = _env_path("CV_OUTPUT_DIR", Path(gcs_to_fuse(_default_output)))

# ---------------------------------------------------------------------------
# Label space
# ---------------------------------------------------------------------------
# Country classes (folder names) that are actually populated in the curated pool.
# Scope of the us-west2 rebuild: classes that can be filled robustly from GLOBE + SAA alone.
#   US/UK/AU/IN = GLOBE (volume) + SAA (speaker diversity); the hybrid label is the country.
#   CN = GLOBE Hong Kong + SAA Mandarin/Cantonese (native-language axis) — the smallest
#        class (lower bound).
# (the NG/JP/CN-only sources AfriSpeech/SpeechOcean762 are not in this bucket — see DATASET.md)
# [2026-07-23] CA removed (6→5 classes): the chronic confusion caused by US/CA accent
# similarity (CA F1 .32~.44 on VoxForge OOD; CA was also the largest drag on mt-v2
# country_macro_f1 — reports/2026-07-18-*, 2026-07-23-mt-v2-a100-sweep.md) was not resolved
# by head/HP/augmentation tuning, so the product decision to remove CA from the country
# label space entirely was reconfirmed (replacing the 2026-07-18 decision to "keep them
# separate"). The curated_spoof/real_fake_5k manifest still contains country="CA" rows,
# but LABEL2ID.get(..., IGNORE) automatically excludes them from the country loss (no
# code change needed, see prepare_data_multitask.py) — they still contribute to the
# real/fake head as real examples.
# Fixed order -> integer label id. Keep stable; the trained head depends on it.
LABELS: list[str] = ["US", "UK", "AU", "IN", "CN"]
LABEL2ID: dict[str, int] = {name: i for i, name in enumerate(LABELS)}
ID2LABEL: dict[int, str] = {i: name for name, i in LABEL2ID.items()}
NUM_LABELS = len(LABELS)

# ---------------------------------------------------------------------------
# Spoof / fake-detection axis (multi-task real/fake head)
# ---------------------------------------------------------------------------
# Training reads the flat, pre-balanced real/fake pool (DATASET.md §11):
#   curated_spoof/real_fake_5k/manifest.csv (label,country,source,system_id,
#   speaker,orig_split,fname,audio_uri) — audio_uri is a full gs:// path (real
#   country-sourced rows point straight at curated/<CC>/audio/, ASVspoof-derived
#   and oversample-dup rows point at real_fake_5k/audio_asv|audio_dup/), so the
#   dataset loader resolves audio directly from audio_uri and never needs a
#   separate root per row. Default is CURATED_ROOT's "sibling" so Vertex's
#   CV_CURATED_ROOT injection auto-resolves this too; override with
#   CV_REAL_FAKE_ROOT.
REAL_FAKE_ROOT = _env_path(
    "CV_REAL_FAKE_ROOT", CURATED_ROOT.parent / "curated_spoof" / "real_fake_5k"
)
# Legacy raw ASVspoof 2019 LA corpus (bonafide/spoof per protocol split). Kept
# only as the read-only source real_fake_5k was built from — training no
# longer reads this directly (DATASET.md §10/§11).
SPOOF_ROOT = _env_path(
    "CV_SPOOF_ROOT", CURATED_ROOT.parent / "curated_spoof" / "asvspoof2019_la"
)
SPOOF_SPLITS: list[str] = ["train", "dev", "eval"]

# Labels of the real/fake head (binary). Fixed order: real=0, fake=1 (the trained fake
# head depends on it).
FAKE_LABELS: list[str] = ["real", "fake"]
FAKE2ID: dict[str, int] = {name: i for i, name in enumerate(FAKE_LABELS)}
ID2FAKE: dict[int, str] = {i: name for name, i in FAKE2ID.items()}
NUM_FAKE_LABELS = len(FAKE_LABELS)

# Sentinel that excludes clips without a country label (= the spoof corpus) from the
# country-head loss. Same convention as torch.nn.functional.cross_entropy(ignore_index=...).
COUNTRY_IGNORE_INDEX = -100

# The precomputed split of real_fake_5k is speaker-level only and does not hide the
# attack type (system_id) — so the fake test metrics are optimistically biased
# (DATASET.md §11). The system tiers below reassign the fake rows so that test.csv itself
# contains genuinely unseen attacks (keeping just the three train/val/test splits, with no
# separate fourth bucket; see prepare_data_multitask.py).
# A01-A06 (original train/dev) and A07-A19 (original eval) have completely separate
# speaker pools (30 and 48 speakers respectively, verified from the manifest), so picking
# both tiers below from A07-A19 only means no speaker leakage with the train-only
# systems (A01-A06).
FAKE_TEST_ONLY_SYSTEMS: frozenset[str] = frozenset({"A17", "A18", "A19"})
FAKE_MIXED_SYSTEMS: frozenset[str] = frozenset({"A16"})
FAKE_MIXED_TEST_FRACTION = 0.33  # share of A16 speakers sent to test (the rest go to train/val)

# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16_000  # sampling rate required by wav2vec2 (16 kHz)
MAX_DURATION_S = 8.0  # crop cap; collator pads to per-batch max up to this
MAX_SAMPLES = int(SAMPLE_RATE * MAX_DURATION_S)  # convert the length in seconds to a number of samples

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
# Large classes are undersampled only down to the per-class cap (TARGET_PER_CLASS). The
# curated pool was rebuilt at 5k scale (2026-07-16, spec_5k.json): US/UK/CA have ~6k,
# AU/IN 4~4.8k, and CN (the smallest class, capped by Hong Kong English) ~1.17k. With the
# cap at 5000 the three large classes are capped at 5000 and the rest are used in full.
# The imbalance is absorbed by macro-F1 + class weighting.
TARGET_PER_CLASS = 5000           # balanced under-sampling cap per class
# Per-speaker clip cap. Some sources (e.g. SpeechOcean762 CN) bundle hundreds of clips
# under one speaker (id); left as-is, a single speaker would dominate a whole class and
# split. Clips per speaker are capped at this value first so that classes/splits are
# filled with diverse speakers.
# (the original curation in DATASET.md also uses a ≤30 clips/speaker convention per source.)
MAX_CLIPS_PER_SPEAKER = 20        # per-speaker clip cap (applied before class cap)
VAL_FRACTION = 0.15               # speaker-level validation holdout fraction
TEST_FRACTION = 0.15              # speaker-level test holdout fraction
SEED = 42                         # fixed random seed for reproducibility

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# Name of the pretrained backbone model to use. Another model can be selected through
# the CV_MODEL_NAME environment variable; the default is facebook/wav2vec2-base.
MODEL_NAME = os.environ.get("CV_MODEL_NAME", "facebook/wav2vec2-base")
