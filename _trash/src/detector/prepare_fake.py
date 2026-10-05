"""Fill data/detector/fake/ round-robin from the six training vocoders.

- LJSpeech utterance IDs are assigned evenly to the six vocoders (no duplicate IDs) -> fake ~= real (13,100), balanced.
- hifiGAN is reserved as an unseen hold-out (excluded here; train.py uses it only to check generalization).
- Link names '<vocoder>__<original file name>' track the origin and prevent collisions.
- fake/ can be regenerated with this script at any time (fixed seed).

Run: .venv/bin/python src/detector/prepare_fake.py
"""
import os
import re
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GEN = PROJECT_ROOT / "data" / "detector" / "generated_audio"
FAKE = PROJECT_ROOT / "data" / "detector" / "fake"

TRAIN_VOCODERS = [
    "ljspeech_full_band_melgan",
    "ljspeech_melgan",
    "ljspeech_melgan_large",
    "ljspeech_multi_band_melgan",
    "ljspeech_parallel_wavegan",
    "ljspeech_waveglow",
]
HOLDOUT = "ljspeech_hifiGAN"  # not placed in fake/ (reserved for the generalization check)

SEED = 42
ID_RE = re.compile(r"^(LJ\d{3}-\d{4})")


def index_vocoder(voc):
    """Index the wav files in a folder as {utterance ID: file name}."""
    m = {}
    for f in os.listdir(GEN / voc):
        if not f.endswith(".wav"):
            continue
        mo = ID_RE.match(f)
        if mo:
            m[mo.group(1)] = f
    return m


def main():
    random.seed(SEED)

    indexes = {v: index_vocoder(v) for v in TRAIN_VOCODERS}
    # utterance IDs present in every training vocoder (intersection)
    common = set.intersection(*(set(ix.keys()) for ix in indexes.values()))
    ids = sorted(common)
    random.shuffle(ids)
    print(f"common utterance IDs: {len(ids)}")

    FAKE.mkdir(parents=True, exist_ok=True)
    # clean up existing links (for reruns)
    for f in os.listdir(FAKE):
        p = FAKE / f
        if p.is_symlink() or p.is_file():
            p.unlink()

    per_voc = {v: 0 for v in TRAIN_VOCODERS}
    for i, uid in enumerate(ids):
        voc = TRAIN_VOCODERS[i % len(TRAIN_VOCODERS)]
        src_name = indexes[voc][uid]
        rel = os.path.join("..", "generated_audio", voc, src_name)  # path relative to fake/
        os.symlink(rel, FAKE / f"{voc}__{src_name}")
        per_voc[voc] += 1

    print(f"\nlinks created in fake/: {sum(per_voc.values())}")
    for v in TRAIN_VOCODERS:
        print(f"  {v}: {per_voc[v]}")
    print(f"\nhold-out (reserved, not in fake): {HOLDOUT}")


if __name__ == "__main__":
    main()
