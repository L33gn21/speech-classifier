import os
from pathlib import Path

import numpy as np
import torch

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset import WaveFakeDataset, utt_id
from model import Detector

# train.py lives at src/detector/train.py -> project root is three levels up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "detector"        # expects real/ and fake/ subdirs
GEN_DIR = DATA_DIR / "generated_audio"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "detector"

# 미학습(hold-out) 보코더: fake/ 에는 들어있지 않고, 일반화 검증에만 쓴다.
HOLDOUT_VOCODER = "ljspeech_hifiGAN"

SEED = 42
TEST_SIZE = 0.2
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4


def list_dir(d, label):
    return [(os.path.join(d, f), label) for f in os.listdir(d)]


def group_split(files, test_size, seed):
    """발화 ID 단위로 train/test 를 나눈다 (같은 ID 의 real/fake 는 같은 쪽으로)."""
    ids = sorted({utt_id(p) for p, _ in files})
    train_ids, test_ids = train_test_split(ids, test_size=test_size, random_state=seed)
    train_ids, test_ids = set(train_ids), set(test_ids)
    train = [(p, y) for p, y in files if utt_id(p) in train_ids]
    test = [(p, y) for p, y in files if utt_id(p) in test_ids]
    return train, test, train_ids, test_ids


@torch.no_grad()
def evaluate(model, loader, device):
    """전체 정확도 + 클래스별 recall(real 을 real 로 / fake 를 fake 로) 반환."""
    model.eval()
    correct = total = 0
    per_class_correct = {0: 0, 1: 0}
    per_class_total = {0: 0, 1: 0}
    for x, y in loader:
        x = x.to(device)
        pred = model(x).argmax(1).cpu()
        for yi, pi in zip(y.tolist(), pred.tolist()):
            per_class_total[yi] += 1
            if yi == pi:
                per_class_correct[yi] += 1
                correct += 1
            total += 1
    acc = correct / total if total else 0.0
    real_recall = per_class_correct[0] / per_class_total[0] if per_class_total[0] else float("nan")
    fake_recall = per_class_correct[1] / per_class_total[1] if per_class_total[1] else float("nan")
    return acc, real_recall, fake_recall


def build_splits():
    """발화 ID 단위 train/test split + hifiGAN hold-out 파일 리스트를 구성.

    train.py(학습)와 eval_holdout.py(평가)가 동일한(seed 고정) split 을 공유하도록
    한 곳에서 만든다. 반환: (train_files, test_files, holdout_files, train_ids, test_ids)
    """
    real_files = list_dir(DATA_DIR / "real", 0)
    fake_files = list_dir(DATA_DIR / "fake", 1)
    all_files = real_files + fake_files

    train_files, test_files, train_ids, test_ids = group_split(all_files, TEST_SIZE, SEED)

    # hifiGAN hold-out: test 쪽 발화 ID 의 real + 미학습 보코더(hifiGAN) fake.
    # -> 미학습 보코더 & 미학습 발화 조합으로 가장 엄격하게 일반화를 측정.
    holdout_real = [(p, y) for p, y in test_files if y == 0]
    holdout_fake = [
        (os.path.join(GEN_DIR, HOLDOUT_VOCODER, f), 1)
        for f in os.listdir(GEN_DIR / HOLDOUT_VOCODER)
        if utt_id(f) in test_ids
    ]
    holdout_files = holdout_real + holdout_fake
    return train_files, test_files, holdout_files, train_ids, test_ids


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_files, test_files, holdout_files, train_ids, test_ids = build_splits()

    print(f"발화 ID: train={len(train_ids)} test={len(test_ids)}")
    print(f"train 파일: {len(train_files)}  test(in-domain) 파일: {len(test_files)}")
    h_real = sum(1 for _, y in holdout_files if y == 0)
    print(f"hifiGAN hold-out 파일: {len(holdout_files)} "
          f"(real={h_real} fake={len(holdout_files) - h_real})")

    train_ds = WaveFakeDataset.from_files(train_files)
    test_ds = WaveFakeDataset.from_files(test_files)
    holdout_ds = WaveFakeDataset.from_files(holdout_files)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    holdout_loader = DataLoader(holdout_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Detector().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_acc = 0.0
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        loss_sum = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", enabled=use_amp):
                pred = model(x)
                loss = criterion(pred, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_sum += loss.item()

        acc, rr, fr = evaluate(model, test_loader, device)
        print(f"epoch {epoch:2d}  loss={loss_sum/len(train_loader):.4f}  "
              f"test acc={acc:.3f} (real={rr:.3f} fake={fr:.3f})")

        # in-domain test 기준 best 를 저장.
        if acc >= best_acc:
            best_acc = acc
            torch.save(model.state_dict(), OUTPUT_DIR / "detector.pt")

    # 학습 종료 후: 미학습 보코더(hifiGAN) 일반화 성능.
    h_acc, h_rr, h_fr = evaluate(model, holdout_loader, device)
    print("\n=== hifiGAN hold-out (미학습 보코더 일반화) ===")
    print(f"acc={h_acc:.3f}  real recall={h_rr:.3f}  fake recall={h_fr:.3f}")
    print(f"(best in-domain test acc={best_acc:.3f}, 저장: {OUTPUT_DIR / 'detector.pt'})")


if __name__ == "__main__":
    main()
