"""Evaluate in-domain test and hifiGAN hold-out performance with the saved detector.pt.

It rebuilds the same (fixed-seed) split as train.py, so it measures generalization to test
utterances not used in training and to an unseen vocoder (hifiGAN) as-is. Used to get just the
baseline numbers of the current checkpoint without rerunning training.

Run: .venv/bin/python src/detector/eval_holdout.py
"""
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import WaveFakeDataset
from model import Detector
from train import build_splits, evaluate, BATCH_SIZE

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WEIGHTS = PROJECT_ROOT / "outputs" / "detector" / "detector.pt"


def main():
    _, test_files, holdout_files, _, test_ids = build_splits()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Detector().to(device)
    model.load_state_dict(torch.load(WEIGHTS, map_location=device))

    test_loader = DataLoader(
        WaveFakeDataset.from_files(test_files),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
    )
    holdout_loader = DataLoader(
        WaveFakeDataset.from_files(holdout_files),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
    )

    print(f"checkpoint: {WEIGHTS}")
    print(f"test (in-domain) files: {len(test_files)}  hifiGAN hold-out files: {len(holdout_files)}\n")

    acc, rr, fr = evaluate(model, test_loader, device)
    print(f"[in-domain test]   acc={acc:.3f}  real recall={rr:.3f}  fake recall={fr:.3f}")

    h_acc, h_rr, h_fr = evaluate(model, holdout_loader, device)
    print(f"[hifiGAN hold-out] acc={h_acc:.3f}  real recall={h_rr:.3f}  fake recall={h_fr:.3f}")
    print("\n(hold-out fake recall = share of unseen-vocoder synthetic clips caught as fake = the key generalization metric)")


if __name__ == "__main__":
    main()
