"""Phase 3 — training.

Fine-tune the accent classifier with HuggingFace Trainer.

Default recipe: freeze the whole wav2vec2 backbone and train only the linear
head. Pass --unfreeze-top N to also fine-tune the top N transformer layers
(do this once the head alone is working).

Outputs are written to config.OUTPUT_DIR, which resolves to:
  - a local dir by default, or
  - AIP_MODEL_DIR (Vertex AI, FUSE-mounted GCS) when running as a Custom Job.

Example (local):
    python train.py --epochs 8 --batch-size 8 --grad-accum 2
    python train.py --unfreeze-top 4 --lr 2e-5 --epochs 6
"""
# 3단계 — 모델 학습.
#
# HuggingFace Trainer를 이용해 억양 분류기를 파인튜닝한다.
#
# 기본 학습 방식: wav2vec2 백본 전체를 동결하고 선형 헤드만 학습한다.
# --unfreeze-top N 옵션을 주면 상위 N개의 트랜스포머 레이어도 함께
# 파인튜닝한다(헤드만으로 잘 동작하는 것을 먼저 확인한 뒤 시도할 것).
#
# 출력은 config.OUTPUT_DIR에 저장되며, 이는 다음 중 하나로 결정된다:
#   - 기본값: 로컬 디렉터리
#   - Vertex AI Custom Job으로 실행 시: AIP_MODEL_DIR(FUSE 마운트된 GCS)
#
# 실행 예시 (로컬):
#     python train.py --epochs 8 --batch-size 8 --grad-accum 2
#     python train.py --unfreeze-top 4 --lr 2e-5 --epochs 6
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from transformers import Trainer, TrainingArguments, Wav2Vec2FeatureExtractor

from config import (
    CURATED_ROOT,
    ID2LABEL,
    LABELS,
    MODEL_NAME,
    OUTPUT_DIR,
    SEED,
    TARGET_PER_CLASS,
    TEST_FRACTION,
    VAL_FRACTION,
)
from dataset import AccentDataset, DataCollator
from model import AccentClassifier
from prepare_data import build_splits, report


def compute_metrics(eval_pred):
    # HuggingFace Trainer가 평가(evaluate) 시마다 호출하는 콜백.
    # 로짓과 정답 레이블을 받아 정확도, 매크로 F1, 클래스별 F1을 계산한다.
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    per_class_f1 = f1_score(labels, preds, average=None, labels=list(range(len(LABELS))))
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        # macro_f1: 클래스 불균형에 덜 민감하도록 클래스별 F1의 단순 평균을 사용.
        # (모델 선택 기준(metric_for_best_model)으로도 사용됨)
        "macro_f1": f1_score(labels, preds, average="macro"),
    }
    for i, name in ID2LABEL.items():
        metrics[f"f1_{name}"] = float(per_class_f1[i])
    return metrics


def detailed_report(preds: np.ndarray, labels: np.ndarray) -> dict:
    """Per-class precision/recall/f1/support + confusion matrix for the model tester.

    ``compute_metrics`` only surfaces the scalars HF Trainer needs for model
    selection (accuracy, macro-F1, per-class F1). The tester dashboard wants a
    richer, per-country breakdown, so from the raw predictions we also compute
    precision/recall/support and the full confusion matrix and stash them in
    ``final_metrics.json`` under nested keys. Keyed by label name so it survives
    label-order changes and is self-describing to the frontend.
    """
    # 모델 테스터 대시보드용 상세 지표: 클래스별 precision/recall/f1/support 와
    # 혼동 행렬. compute_metrics 는 모델 선택에 필요한 스칼라만 내보내므로,
    # 여기서 원본 예측으로부터 나라별 상세치를 추가로 계산해 final_metrics.json 에
    # 중첩 키로 저장한다(프론트가 그대로 시각화).
    ids = list(range(len(LABELS)))
    p, r, f1, support = precision_recall_fscore_support(
        labels, preds, labels=ids, zero_division=0
    )
    per_class = {
        LABELS[i]: {
            "precision": float(p[i]),
            "recall": float(r[i]),        # 대각선 재현율 = 그 나라 클립을 맞춘 비율
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in ids
    }
    cm = confusion_matrix(labels, preds, labels=ids)
    return {"labels": LABELS, "per_class": per_class, "confusion_matrix": cm.tolist()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=8.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--unfreeze-top", type=int, default=0,
                    help="unfreeze top N transformer layers (0 = head only)")
                    # 상위 N개 트랜스포머 레이어를 학습 가능하게 해제 (0이면 헤드만 학습)
    ap.add_argument("--gradient-checkpointing", action="store_true")
    # 메모리 절약을 위한 그래디언트 체크포인팅 활성화 옵션(속도 대신 메모리 절약).
    ap.add_argument("--output-dir", default=str(OUTPUT_DIR))
    ap.add_argument("--curated-root", default=str(CURATED_ROOT))
    ap.add_argument("--per-class", type=int, default=TARGET_PER_CLASS)
    ap.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    ap.add_argument("--test-fraction", type=float, default=TEST_FRACTION)
    ap.add_argument("--no-fp16", action="store_true")
    # 기본은 fp16(반정밀도) 학습, 이 플래그로 비활성화 가능 (예: CPU 학습 시).
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Vertex AI Custom Jobs configured with a TensorBoard resource + service
    # account inject AIP_TENSORBOARD_LOG_DIR (a GCS path) and continuously sync
    # anything written there to the TensorBoard instance while the job runs.
    # Falls back to a local dir for plain (non-Vertex) runs.
    tb_log_dir = os.environ.get(
        "AIP_TENSORBOARD_LOG_DIR", os.path.join(args.output_dir, "tb_logs")
    )

    # curated 매니페스트로부터 화자 단위 train/val/test 분할을 그 자리에서 생성.
    # 원본 curated/ 는 읽기만 하고, 분할 결과 CSV는 output_dir 아래에만 기록한다.
    train_df, val_df, test_df = build_splits(
        curated_root=args.curated_root,
        per_class=args.per_class,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=SEED,
    )
    report("train", train_df)
    report("val", val_df)
    report("test", test_df)
    manifest_out = os.path.join(args.output_dir, "manifests")
    os.makedirs(manifest_out, exist_ok=True)
    cols = ["filename", "label", "country", "speaker", "source"]
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        part[cols].to_csv(os.path.join(manifest_out, f"{name}.csv"), index=False)

    # Wav2Vec2 입력 전처리기(정규화 담당)와, 이를 사용하는 배치 콜레이터 준비.
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    collator = DataCollator(feature_extractor)

    train_ds = AccentDataset(train_df, curated_root=args.curated_root)
    eval_ds = AccentDataset(val_df, curated_root=args.curated_root)
    test_ds = AccentDataset(test_df, curated_root=args.curated_root)
    print(f"train={len(train_ds)}  val={len(eval_ds)}  test={len(test_ds)}")

    model = AccentClassifier(MODEL_NAME)
    if args.unfreeze_top > 0:
        # 백본의 상위 N개 레이어까지 함께 파인튜닝하는 모드.
        model.unfreeze_top_layers(args.unfreeze_top)
        print(f"backbone frozen except top {args.unfreeze_top} transformer layers")
    else:
        # 기본 모드: 백본은 완전히 동결하고 헤드만 학습(가장 빠르고 안전한 시작점).
        model.freeze_backbone()
        print("backbone fully frozen (training head only)")
    # 학습 가능한 파라미터 수 대비 전체 파라미터 수를 로그로 남겨 확인.
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable:,} / {total:,}")

    # --- diagnostic: do pooled logits actually vary across different clips? ---
    # 서로 다른 클립이 서로 다른 pooled 출력을 내는지 확인. batch-std 가 0에 가까우면
    # 피처가 붕괴(모든 입력이 사실상 같은 표현)한 것이고, 그러면 헤드가 클래스를
    # 분리할 수 없어 손실이 ln(클래스수)에 갇힌다. 학습 시작 전 1회만 찍는다.
    import torch as _torch
    _n = min(16, len(train_ds))
    if _n >= 2:
        model.eval()
        with _torch.no_grad():
            _b = collator([train_ds[i] for i in range(_n)])
            _out = model(input_values=_b["input_values"],
                         attention_mask=_b.get("attention_mask"))
            _lg = _out.logits  # [n, C]
            _bstd = float(_lg.std(dim=0).mean())   # variation ACROSS clips (want > 0)
            _lbl = [int(train_ds[i]["label"]) for i in range(_n)]
            print(f"[diag] pooled logits {tuple(_lg.shape)}  across-clip std={_bstd:.4f}  "
                  f"pred={_lg.argmax(-1).tolist()}  true={_lbl}")
        model.train()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        fp16=not args.no_fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        # use_reentrant=False so checkpointing works when lower backbone layers
        # are frozen (reentrant mode errors when no checkpoint input needs grad).
        # 하위 백본 레이어들이 동결되어 있을 때(즉 체크포인트 구간의 입력이
        # grad를 필요로 하지 않을 때) reentrant 모드는 에러를 낸다.
        # use_reentrant=False로 설정해 이 문제를 회피한다.
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch",   # 매 에포크마다 평가 수행
        save_strategy="epoch",   # 매 에포크마다 체크포인트 저장
        logging_steps=50,
        load_best_model_at_end=True,       # 학습 종료 시 가장 좋은 체크포인트를 로드
        metric_for_best_model="macro_f1",  # "가장 좋다"의 기준은 매크로 F1
        greater_is_better=True,
        save_total_limit=2,   # 디스크 절약을 위해 최근 2개 체크포인트만 보관
        dataloader_num_workers=4,
        remove_unused_columns=False,  # our model consumes raw batch dict
        # HF Trainer는 기본적으로 모델 forward 시그니처에 없는 배치 컬럼을 자동
        # 제거하는데, 우리 모델은 콜레이터가 만든 배치 딕셔너리를 그대로
        # 소비하므로 이 자동 제거 기능을 꺼야 한다.
        report_to=["tensorboard"],
        logging_dir=tb_log_dir,  # Vertex AI syncs this to the linked TensorBoard instance
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    # predict() (evaluate() 대신) 를 써서 스칼라 지표와 "원본 예측"을 함께 얻는다.
    # 원본 예측이 있어야 혼동 행렬·클래스별 precision/recall 을 계산할 수 있다.
    val_out = trainer.predict(eval_ds, metric_key_prefix="eval")
    val_metrics = val_out.metrics
    print("final val eval:", json.dumps(val_metrics, indent=2))
    # 학습에 전혀 쓰이지 않은 홀드아웃 test 셋으로 최종 성능도 측정.
    test_out = trainer.predict(test_ds, metric_key_prefix="test")
    test_metrics = test_out.metrics
    print("final test eval:", json.dumps(test_metrics, indent=2))
    metrics = {**val_metrics, **test_metrics}
    # 나라별 상세 지표 + 혼동 행렬을 (val/test 각각) 중첩 키로 함께 저장한다.
    metrics["eval_detail"] = detailed_report(
        np.argmax(val_out.predictions, axis=-1), val_out.label_ids)
    metrics["test_detail"] = detailed_report(
        np.argmax(test_out.predictions, axis=-1), test_out.label_ids)

    # persist head + backbone weights, feature extractor, and label config
    # 학습된 헤드+백본 가중치, feature extractor 설정, 레이블 매핑 정보를
    # 모두 output_dir에 저장하여 나중에 evaluate.py / infer.py가 그대로
    # 불러올 수 있게 한다.
    trainer.save_model(args.output_dir)
    feature_extractor.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "label_config.json"), "w") as f:
        json.dump({"labels": LABELS, "id2label": ID2LABEL}, f, indent=2)
    with open(os.path.join(args.output_dir, "final_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"saved to {args.output_dir}")


if __name__ == "__main__":
    main()
