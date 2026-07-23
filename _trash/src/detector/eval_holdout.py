"""저장된 detector.pt 로 in-domain test 와 hifiGAN hold-out 성능을 평가한다.

train.py 와 동일한(seed 고정) split 을 재구성하므로, 학습에 쓰지 않은 test 발화와
미학습 보코더(hifiGAN)에 대한 일반화를 그대로 측정한다. 학습을 다시 돌리지 않고
현재 체크포인트의 baseline 수치만 뽑을 때 사용.

실행: .venv/bin/python src/detector/eval_holdout.py
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

    print(f"체크포인트: {WEIGHTS}")
    print(f"test(in-domain) 파일: {len(test_files)}  hifiGAN hold-out 파일: {len(holdout_files)}\n")

    acc, rr, fr = evaluate(model, test_loader, device)
    print(f"[in-domain test]   acc={acc:.3f}  real recall={rr:.3f}  fake recall={fr:.3f}")

    h_acc, h_rr, h_fr = evaluate(model, holdout_loader, device)
    print(f"[hifiGAN hold-out] acc={h_acc:.3f}  real recall={h_rr:.3f}  fake recall={h_fr:.3f}")
    print("\n(hold-out fake recall = 미학습 보코더 합성음을 fake 로 잡아내는 비율 = 일반화 핵심 지표)")


if __name__ == "__main__":
    main()
