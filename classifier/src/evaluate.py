"""Phase 3 — testing.

Evaluate a trained accent classifier on the held-out test manifest and print
accuracy, macro-F1, per-class F1, and a confusion matrix. Speaker-disjoint by
construction (see prepare_data.py), so these numbers reflect accent, not voice.

Example:
    python evaluate.py --model-dir ../outputs/classifier
"""
# 3단계 — 테스트(평가).
#
# 학습된 억양 분류기를 홀드아웃(held-out) 테스트 매니페스트에서 평가하여
# 정확도, 매크로 F1, 클래스별 F1, 혼동 행렬(confusion matrix)을 출력한다.
# prepare_data.py에서 이미 화자 단위로 분리했기 때문에(speaker-disjoint),
# 이 지표들은 "목소리를 외운 결과"가 아니라 실제 억양 구분 성능을 반영한다.
#
# 실행 예시:
#     python evaluate.py --model-dir ../outputs/classifier
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor

from config import CURATED_ROOT, LABELS, OUTPUT_DIR
from dataset import AccentDataset, DataCollator
from model import AccentClassifier, load_from_dir


def load_trained(model_dir: str) -> AccentClassifier:
    # 저장된 모델 디렉터리에서 가중치를 불러와 평가 모드(eval)의 모델을 반환.
    # load_from_dir 가 model_config.json 을 읽어 학습 때와 동일한 구조(백본·헤드·
    # 레이어가중)로 골격을 만든다(구버전은 레거시 기본값으로 폴백).
    model = load_from_dir(model_dir)
    safepath = os.path.join(model_dir, "model.safetensors")
    binpath = os.path.join(model_dir, "pytorch_model.bin")
    if os.path.exists(safepath):
        # safetensors 포맷을 우선 시도(더 안전하고 빠름).
        from safetensors.torch import load_file

        state = load_file(safepath)
    elif os.path.exists(binpath):
        # 없으면 구형 pytorch .bin 포맷으로 폴백.
        state = torch.load(binpath, map_location="cpu")
    else:
        raise FileNotFoundError(f"no weights in {model_dir}")
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(OUTPUT_DIR))
    # 학습 잡이 model-dir/manifests 아래에 test.csv 를 함께 저장한다.
    ap.add_argument("--manifest-dir", default=None,
                    help="dir holding test.csv (default: <model-dir>/manifests)")
    ap.add_argument("--curated-root", default=str(CURATED_ROOT))
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    manifest_dir = args.manifest_dir or os.path.join(args.model_dir, "manifests")

    # GPU가 있으면 GPU를 사용, 없으면 CPU로 자동 전환.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_trained(args.model_dir).to(device)
    fe = AutoFeatureExtractor.from_pretrained(args.model_dir)
    collator = DataCollator(fe)

    test_ds = AccentDataset(os.path.join(manifest_dir, "test.csv"),
                            curated_root=args.curated_root)
    loader = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=collator)
    print(f"test={len(test_ds)} clips on {device}")

    all_preds, all_labels = [], []
    for batch in loader:
        # labels는 모델 forward에 넘기지 않고 따로 빼서 나중에 비교용으로 사용.
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        all_preds.append(logits.argmax(-1).cpu().numpy())
        all_labels.append(labels.numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    ids = list(range(len(LABELS)))
    print(f"\naccuracy : {accuracy_score(labels, preds):.4f}")
    print(f"macro_f1 : {f1_score(labels, preds, average='macro', labels=ids):.4f}\n")
    # sklearn의 상세 리포트: 클래스별 precision/recall/F1/지원 개수(support).
    print(classification_report(labels, preds, labels=ids, target_names=LABELS, digits=4))
    print("confusion matrix (rows=true, cols=pred):")
    # 혼동 행렬: 행=실제 정답 클래스, 열=예측 클래스.
    # 대각선이 정답을 맞춘 개수, 그 외는 어느 클래스로 잘못 예측했는지를 보여줌.
    cm = confusion_matrix(labels, preds, labels=ids)
    header = "          " + "".join(f"{l:>10s}" for l in LABELS)
    print(header)
    for i, row in enumerate(cm):
        print(f"{LABELS[i]:>10s}" + "".join(f"{v:10d}" for v in row))

    # 평가 결과를 JSON으로도 저장해 나중에 프로그램적으로 참조할 수 있게 함.
    out = {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average='macro', labels=ids)),
        "confusion_matrix": cm.tolist(),
    }
    dest = os.path.join(args.model_dir, "test_report.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {dest}")


if __name__ == "__main__":
    main()
