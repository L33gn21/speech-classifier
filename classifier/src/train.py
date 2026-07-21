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
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from transformers import (
    AutoFeatureExtractor,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

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
from model import AccentClassifier, write_model_config
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


def compute_class_weights(train_df, scheme: str):
    """Per-class loss weights from the *train* split label counts.

    Kept out of the model so the saved state_dict is unchanged (evaluate.py /
    infer.py / model_tester reload identical weights). All schemes renormalize
    to mean weight ~= 1 (so the effective learning rate is unchanged):
      - ``balanced``: w ∝ 1/count — full inverse-frequency (equals sklearn's
        "balanced"). Fully compensates the imbalance; strongest push on CN.
      - ``sqrt``: w ∝ 1/sqrt(count) — tempered; a gentler middle ground that
        lifts minority classes without over-emphasizing the rarest one.
    Returns None for ``none`` (plain, unweighted cross-entropy).
    """
    # 학습(train) 분할의 클래스별 클립 수로부터 손실 가중치를 만든다. 모델이 아니라
    # 여기(WeightedTrainer)에만 얹으므로 저장되는 가중치(state_dict)는 그대로 유지되어
    # evaluate.py / infer.py / model_tester 가 동일하게 로드된다. CN(최소 클래스) 등
    # 소수 클래스의 손실 기여를 키워 macro-F1·소수 클래스 recall 을 끌어올린다.
    if scheme == "none":
        return None
    counts = (
        train_df["label"].value_counts().reindex(range(len(LABELS)), fill_value=0)
        .to_numpy(dtype=np.float64)
    )
    counts = np.clip(counts, 1.0, None)  # avoid div-by-zero for an empty class
    if scheme == "balanced":
        w = 1.0 / counts                 # full inverse-frequency
    elif scheme == "sqrt":
        w = 1.0 / np.sqrt(counts)        # tempered — gentler on the rarest class
    else:
        raise ValueError(f"unknown class-weight scheme: {scheme}")
    w = w * (len(LABELS) / w.sum())  # normalize so mean weight ~= 1
    return torch.tensor(w, dtype=torch.float32)


class WeightedTrainer(Trainer):
    """HF Trainer with an optional class-weighted cross-entropy loss.

    The model's ``forward`` still returns its own (unweighted) loss, but the
    Trainer selects the loss via ``compute_loss`` — so we recompute a weighted
    cross-entropy from the logits here and ignore the model's. This keeps the
    weighting entirely on the training side; nothing about the saved model
    changes. ``class_weights=None`` reproduces the previous plain-CE behavior.
    """
    # 클래스 가중 교차엔트로피를 적용하는 Trainer. 모델의 forward 는 여전히 자체
    # (비가중) 손실을 계산하지만, Trainer 는 compute_loss 로 손실을 고른다 — 여기서
    # 로짓으로부터 가중 CE 를 다시 계산해 그것을 쓴다. 가중치는 학습 쪽에만 존재하고
    # 저장 모델은 그대로다. class_weights=None 이면 기존 plain-CE 와 동일.
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights

    # num_items_in_batch / **kwargs: transformers>=4.46 passes extra kwargs to
    # compute_loss. We pin 4.44.2 (which doesn't), but accept-and-ignore them so
    # a future pin bump can't silently break the training step.
    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = None
        if self._class_weights is not None:
            weight = self._class_weights.to(logits.device)
        loss = F.cross_entropy(logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=8.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    # Underscore aliases (--warmup_ratio, --weight_decay, --unfreeze_top) are added
    # so Vertex AI Vizier — which injects each trial's params as --<parameterId>=v
    # with underscores — matches these flags without a separate mapping.
    ap.add_argument("--warmup-ratio", "--warmup_ratio", dest="warmup_ratio",
                    type=float, default=0.1)
    ap.add_argument("--weight-decay", "--weight_decay", dest="weight_decay",
                    type=float, default=0.01)
    ap.add_argument("--unfreeze-top", "--unfreeze_top", dest="unfreeze_top",
                    type=int, default=0,
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
    ap.add_argument("--class-weight", "--class_weight", dest="class_weight",
                    choices=["none", "balanced", "sqrt"], default="balanced",
                    help="per-class loss weighting: none | balanced (1/count) | "
                         "sqrt (tempered) (default: balanced)")
    # 클래스 불균형(US/UK/CA ~6k vs CN ~1.17k)을 손실 가중치로 흡수한다.
    ap.add_argument("--dropout", type=float, default=0.1,
                    help="classifier head dropout")
    ap.add_argument("--early-stopping-patience", type=int, default=3,
                    help="stop if macro_f1 hasn't improved for N evals (0=disable)")
    # macro-F1 이 N번 평가 동안 개선되지 않으면 조기 종료(0이면 비활성화).
    ap.add_argument("--augment", action="store_true",
                    help="light waveform augmentation (gain + noise) on the train split")
    # 학습 분할에만 경량 파형 증강(랜덤 게인 + 가우시안 노이즈)을 적용해 채널
    # confound(GLOBE vs SAA) 에 대한 강건성을 높인다.
    ap.add_argument("--aug-strength", "--aug_strength", dest="aug_strength",
                    type=float, default=0.0,
                    help="with --augment, use domain-randomization augmentation at "
                         "this strength (0=legacy light aug; ~1.0=full). Simulates "
                         "the GLOBE->VoxForge recording-domain shift (speed/band-limit/"
                         "reverb/colored-noise) to fight source-domain overfitting.")
    # --augment 와 함께 쓸 때 도메인 랜덤화 증강 세기. 0 이면 레거시 경량 증강,
    # ~1.0 이면 풀 세기. GLOBE→VoxForge 녹음 도메인 시프트(속도·대역제한·잔향·컬러노이즈)
    # 를 흉내내 소스 도메인 과적합을 줄인다(대책 C, CA→US 붕괴의 도메인시프트 절반 겨냥).
    ap.add_argument("--save-total-limit", "--save_total_limit", dest="save_total_limit",
                    type=int, default=2,
                    help="max checkpoints to keep (default 2). Raise to keep every "
                         "epoch for a checkpoint scan (e.g. OOD-optimal early-stopping).")
    # 보관할 체크포인트 최대 개수(기본 2). 매 epoch 체크포인트를 남겨 OOD(VoxForge) 최적
    # 정지점을 스캔하려면 크게 준다(예: epochs 이상).
    ap.add_argument("--hypertune", action="store_true",
                    help="report eval_macro_f1 to Vertex AI Vizier (HP tuning jobs)")
    # Vertex AI Hyperparameter Tuning(Vizier) 잡에서 trial 점수를 보고할 때만 켠다.
    # --- architecture knobs (v3 experiments) ---------------------------------
    ap.add_argument("--backbone", "--model-name", "--model_name", dest="backbone",
                    default=MODEL_NAME,
                    help="pretrained backbone (e.g. facebook/wav2vec2-base or "
                         "microsoft/wavlm-base-plus)")
    # 백본 교체용. AutoModel 이 이름으로 wav2vec2/wavlm 을 자동 선택한다.
    ap.add_argument("--head", "--head_type", dest="head",
                    choices=["mean", "attentive"], default="mean",
                    help="utterance pooling head: mean (masked mean) | attentive "
                         "(attentive statistics pooling: weighted mean+std)")
    # 발화 풀링 헤드 선택: mean(마스킹 평균) | attentive(어텐션 가중 평균+표준편차).
    ap.add_argument("--layer-weighting", "--layer_weighting", dest="layer_weighting",
                    action="store_true",
                    help="learned weighted sum over all backbone layers (SUPERB-style)")
    # 전 백본 레이어 은닉상태의 학습가능 가중합을 표현으로 사용(SUPERB식).
    ap.add_argument("--mask-time-prob", "--mask_time_prob", dest="mask_time_prob",
                    type=float, default=None,
                    help="SpecAugment time-mask prob (None=backbone default ~0.05)")
    ap.add_argument("--mask-feature-prob", "--mask_feature_prob", dest="mask_feature_prob",
                    type=float, default=None,
                    help="SpecAugment feature-mask prob (None=backbone default)")
    # 백본 내장 SpecAugment 세기(학습 시 특징 시간/채널 마스킹). None 이면 백본 기본값.
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

    # 입력 전처리기(정규화 담당)와, 이를 사용하는 배치 콜레이터 준비. AutoFeatureExtractor
    # 라 wav2vec2/wavlm 어느 백본이든 맞는 전처리기를 자동으로 불러온다.
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.backbone)
    collator = DataCollator(feature_extractor)

    train_ds = AccentDataset(train_df, curated_root=args.curated_root,
                             augment=args.augment, aug_strength=args.aug_strength)
    eval_ds = AccentDataset(val_df, curated_root=args.curated_root)
    test_ds = AccentDataset(test_df, curated_root=args.curated_root)
    aug_kind = ("domain" if args.augment and args.aug_strength > 0
                else "legacy" if args.augment else "off")
    print(f"train={len(train_ds)}  val={len(eval_ds)}  test={len(test_ds)}"
          f"  augment={args.augment}  aug={aug_kind}(strength={args.aug_strength})")

    model = AccentClassifier(args.backbone, dropout=args.dropout,
                             head=args.head, layer_weighting=args.layer_weighting)
    print(f"backbone={args.backbone}  head={args.head}  "
          f"layer_weighting={args.layer_weighting}")
    if args.mask_time_prob is not None or args.mask_feature_prob is not None:
        # 백본 내장 SpecAugment 세기를 조절(학습 시에만 적용됨).
        model.set_spec_augment(args.mask_time_prob, args.mask_feature_prob)
        print(f"spec-augment: mask_time_prob={args.mask_time_prob} "
              f"mask_feature_prob={args.mask_feature_prob}")
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
        save_total_limit=args.save_total_limit,   # 기본 2(디스크 절약). 스캔 시 상향.
        dataloader_num_workers=4,
        remove_unused_columns=False,  # our model consumes raw batch dict
        # HF Trainer는 기본적으로 모델 forward 시그니처에 없는 배치 컬럼을 자동
        # 제거하는데, 우리 모델은 콜레이터가 만든 배치 딕셔너리를 그대로
        # 소비하므로 이 자동 제거 기능을 꺼야 한다.
        report_to=["tensorboard"],
        logging_dir=tb_log_dir,  # Vertex AI syncs this to the linked TensorBoard instance
        seed=SEED,
    )

    # class-weighted loss: computed from the train split so it reflects the
    # actual (post-undersampling) balance the model sees this run.
    class_weights = compute_class_weights(train_df, args.class_weight)
    if class_weights is not None:
        print("class weights (%s): %s" % (
            args.class_weight,
            {LABELS[i]: round(float(class_weights[i]), 3) for i in range(len(LABELS))}))

    callbacks = []
    if args.early_stopping_patience and args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=callbacks,
    )

    trainer.train()
    # predict() (evaluate() 대신) 를 써서 스칼라 지표와 "원본 예측"을 함께 얻는다.
    # 원본 예측이 있어야 혼동 행렬·클래스별 precision/recall 을 계산할 수 있다.
    val_out = trainer.predict(eval_ds, metric_key_prefix="eval")
    val_metrics = val_out.metrics
    print("final val eval:", json.dumps(val_metrics, indent=2))
    # Vertex AI Hyperparameter Tuning(Vizier)에 이 trial 의 점수를 보고한다.
    # --hypertune 일 때만, 그리고 cloudml-hypertune 이 설치돼 있을 때만 동작한다.
    if args.hypertune:
        try:
            import hypertune

            hpt = hypertune.HyperTune()
            hpt.report_hyperparameter_tuning_metric(
                hyperparameter_metric_tag="macro_f1",
                metric_value=float(val_metrics.get("eval_macro_f1", 0.0)),
            )
            print("reported macro_f1 to hypertune:",
                  val_metrics.get("eval_macro_f1"))
        except Exception as e:  # noqa: BLE001 — never fail the job over reporting
            print("hypertune report skipped:", e)
    # 학습에 전혀 쓰이지 않은 홀드아웃 test 셋으로 최종 성능도 측정.
    test_out = trainer.predict(test_ds, metric_key_prefix="test")
    test_metrics = test_out.metrics
    print("final test eval:", json.dumps(test_metrics, indent=2))
    metrics = {**val_metrics, **test_metrics}
    # 재현/추적용으로 이 잡의 증강 설정을 함께 남긴다(추가 키라 대시보드 스키마에 안전).
    metrics["train_config"] = {
        "augment": bool(args.augment),
        "aug_strength": float(args.aug_strength),
        "aug_kind": aug_kind,
        "backbone": args.backbone,
        "head": args.head,
        "unfreeze_top": args.unfreeze_top,
        "per_class": args.per_class,
        "epochs": args.epochs,
    }
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
    # 추론 측(infer/evaluate/model_tester)이 저장 가중치와 동일한 구조로 모델을
    # 재구성할 수 있도록 아키텍처 하이퍼파라미터를 함께 남긴다.
    write_model_config(
        args.output_dir, backbone=args.backbone, num_labels=len(LABELS),
        dropout=args.dropout, head=args.head, layer_weighting=args.layer_weighting)
    with open(os.path.join(args.output_dir, "final_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"saved to {args.output_dir}")


if __name__ == "__main__":
    main()
