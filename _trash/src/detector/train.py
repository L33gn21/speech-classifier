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

# Unseen (hold-out) vocoder: not included in fake/, used only to check generalization.
HOLDOUT_VOCODER = "ljspeech_hifiGAN"

SEED = 42
TEST_SIZE = 0.2
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4


def list_dir(d, label):
    return [(os.path.join(d, f), label) for f in os.listdir(d)]


def group_split(files, test_size, seed):
    """Split train/test by utterance ID (real/fake of the same ID go to the same side)."""
    ids = sorted({utt_id(p) for p, _ in files})
    train_ids, test_ids = train_test_split(ids, test_size=test_size, random_state=seed)
    train_ids, test_ids = set(train_ids), set(test_ids)
    train = [(p, y) for p, y in files if utt_id(p) in train_ids]
    test = [(p, y) for p, y in files if utt_id(p) in test_ids]
    return train, test, train_ids, test_ids


@torch.no_grad()
def evaluate(model, loader, device):
    """Return overall accuracy + per-class recall (real as real / fake as fake)."""
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
    """Build the utterance-ID-level train/test split + the hifiGAN hold-out file list.

    Built in one place so that train.py (training) and eval_holdout.py (evaluation) share the
    same (fixed-seed) split. Returns: (train_files, test_files, holdout_files, train_ids, test_ids)
    """
    real_files = list_dir(DATA_DIR / "real", 0)
    fake_files = list_dir(DATA_DIR / "fake", 1)
    all_files = real_files + fake_files

    train_files, test_files, train_ids, test_ids = group_split(all_files, TEST_SIZE, SEED)

    # hifiGAN hold-out: real clips of the test-side utterance IDs + fakes from the unseen vocoder (hifiGAN).
    # -> the strictest generalization measurement: unseen vocoder & unseen utterances combined.
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

    print(f"utterance IDs: train={len(train_ids)} test={len(test_ids)}")
    print(f"train files: {len(train_files)}  test (in-domain) files: {len(test_files)}")
    h_real = sum(1 for _, y in holdout_files if y == 0)
    print(f"hifiGAN hold-out files: {len(holdout_files)} "
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

        # Save the best model by in-domain test accuracy.
        if acc >= best_acc:
            best_acc = acc
            torch.save(model.state_dict(), OUTPUT_DIR / "detector.pt")

    # After training: generalization to the unseen vocoder (hifiGAN).
    h_acc, h_rr, h_fr = evaluate(model, holdout_loader, device)
    print("\n=== hifiGAN hold-out (unseen-vocoder generalization) ===")
    print(f"acc={h_acc:.3f}  real recall={h_rr:.3f}  fake recall={h_fr:.3f}")
    print(f"(best in-domain test acc={best_acc:.3f}, saved: {OUTPUT_DIR / 'detector.pt'})")


if __name__ == "__main__":
    main()
