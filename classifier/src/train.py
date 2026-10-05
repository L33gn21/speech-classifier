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
    COUNTRY_IGNORE_INDEX,
    CURATED_ROOT,
    FAKE_LABELS,
    ID2FAKE,
    ID2LABEL,
    LABELS,
    MODEL_NAME,
    NUM_FAKE_LABELS,
    OUTPUT_DIR,
    REAL_FAKE_ROOT,
    SEED,
    TEST_FRACTION,
    VAL_FRACTION,
)
from dataset import AccentDataset, DataCollator, MultiTaskCollator, MultiTaskDataset
from model import AccentClassifier, write_model_config
from prepare_data import build_splits, report
from prepare_data_multitask import build_multitask_splits
from prepare_data_multitask import report as report_mt


def compute_metrics(eval_pred):
    # Callback the HuggingFace Trainer calls on every evaluation.
    # Takes the logits and gold labels and computes accuracy, macro-F1 and per-class F1.
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    per_class_f1 = f1_score(labels, preds, average=None, labels=list(range(len(LABELS))))
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        # macro_f1: the plain mean of per-class F1, so it is less sensitive to class imbalance.
        # (also used as the model-selection criterion, metric_for_best_model)
        "macro_f1": f1_score(labels, preds, average="macro"),
    }
    for i, name in ID2LABEL.items():
        metrics[f"f1_{name}"] = float(per_class_f1[i])
    return metrics


def detailed_report(preds: np.ndarray, labels: np.ndarray,
                    label_names: list[str] | None = None) -> dict:
    """Per-class precision/recall/f1/support + confusion matrix for the model tester.

    ``compute_metrics`` only surfaces the scalars HF Trainer needs for model
    selection (accuracy, macro-F1, per-class F1). The tester dashboard wants a
    richer, per-country breakdown, so from the raw predictions we also compute
    precision/recall/support and the full confusion matrix and stash them in
    ``final_metrics.json`` under nested keys. Keyed by label name so it survives
    label-order changes and is self-describing to the frontend.
    """
    # label_names: defaults to the country LABELS. Exposed so it can be reused with other
    # label sets such as the multitask fake head (backward-compatible default; callers
    # need no change).
    names = label_names if label_names is not None else LABELS
    ids = list(range(len(names)))
    p, r, f1, support = precision_recall_fscore_support(
        labels, preds, labels=ids, zero_division=0
    )
    per_class = {
        names[i]: {
            "precision": float(p[i]),
            "recall": float(r[i]),        # diagonal recall = share of that country's clips classified correctly
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in ids
    }
    cm = confusion_matrix(labels, preds, labels=ids)
    return {"labels": names, "per_class": per_class, "confusion_matrix": cm.tolist()}


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


# ===========================================================================
# Multi-task (country + real/fake) training
# ===========================================================================
def _weights_from_counts(counts: np.ndarray, scheme: str, n: int):
    """Inverse-frequency (or sqrt-tempered) class weights, mean-normalized to ~1."""
    # Mean-normalizing keeps the effective learning rate unchanged (same convention as
    # compute_class_weights).
    counts = np.clip(counts.astype(np.float64), 1.0, None)
    if scheme == "balanced":
        w = 1.0 / counts
    elif scheme == "sqrt":
        w = 1.0 / np.sqrt(counts)
    else:
        raise ValueError(f"unknown class-weight scheme: {scheme}")
    w = w * (n / w.sum())
    return torch.tensor(w, dtype=torch.float32)


def compute_country_weights_mt(train_df, scheme: str):
    """Country-head class weights from accent rows only (country_label != -100)."""
    if scheme == "none":
        return None
    lab = train_df.loc[train_df["country_label"] != COUNTRY_IGNORE_INDEX, "country_label"]
    counts = (lab.value_counts().reindex(range(len(LABELS)), fill_value=0)
              .to_numpy(dtype=np.float64))
    return _weights_from_counts(counts, scheme, len(LABELS))


def compute_fake_weights_mt(train_df, scheme: str):
    """Real/fake-head class weights over all clips (absorbs the ~8.7:1 imbalance)."""
    if scheme == "none":
        return None
    counts = (train_df["fake_label"].value_counts().reindex(range(NUM_FAKE_LABELS), fill_value=0)
              .to_numpy(dtype=np.float64))
    return _weights_from_counts(counts, scheme, NUM_FAKE_LABELS)


def compute_metrics_multitask(eval_pred):
    """Metrics for both heads. predictions=(country_logits, fake_logits),
    label_ids=(country_labels, fake_labels) — see MultiTaskCollator + label_names.
    """
    preds, labels = eval_pred.predictions, eval_pred.label_ids
    country_logits, fake_logits = preds[0], preds[1]
    country_labels, fake_labels = labels[0], labels[1]

    metrics = {}
    # --- real/fake head (primary; model selection uses fake_macro_f1) ---
    fp = np.argmax(fake_logits, axis=-1)
    fake_ids = list(range(NUM_FAKE_LABELS))
    metrics["fake_accuracy"] = float(accuracy_score(fake_labels, fp))
    metrics["fake_macro_f1"] = float(f1_score(fake_labels, fp, average="macro", labels=fake_ids))
    f_f1 = f1_score(fake_labels, fp, average=None, labels=fake_ids)
    for i, name in ID2FAKE.items():
        metrics[f"fake_f1_{name}"] = float(f_f1[i])

    # --- country head (only rows with a real country label; spoof rows ignored) ---
    mask = country_labels != COUNTRY_IGNORE_INDEX
    if mask.any():
        cp = np.argmax(country_logits[mask], axis=-1)
        cl = country_labels[mask]
        ids = list(range(len(LABELS)))
        metrics["country_accuracy"] = float(accuracy_score(cl, cp))
        metrics["country_macro_f1"] = float(f1_score(cl, cp, average="macro", labels=ids))

    # --- combined selection metric ---------------------------------------------
    # fake dev(=seen attacks A01-A06) saturates by ~epoch 1, so selecting on
    # fake_macro_f1 alone would let early-stopping cut country short. Select on the
    # mean of both heads so the near-flat-high fake term keeps the model honest
    # while the country term (which actually improves over epochs) drives the pick.
    metrics["mt_macro_f1"] = float(
        (metrics["fake_macro_f1"] + metrics.get("country_macro_f1", metrics["fake_macro_f1"]))
        / 2.0)
    return metrics


class MultiTaskTrainer(Trainer):
    """HF Trainer with a combined country + real/fake weighted loss.

    ``loss = country_CE(ignore_index=-100, weighted) + λ · fake_CE(weighted)``.
    The country head only learns from accent clips (spoof clips carry
    country_label=-100 and are ignored). Reads the two label tensors from the
    batch WITHOUT mutating it (prediction_step re-reads them for compute_metrics),
    and calls the model with the label keys stripped so ``forward`` stays clean.
    """
    def __init__(self, *args, country_weights=None, fake_weights=None,
                 fake_loss_weight: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._cw = country_weights
        self._fw = fake_weights
        self._lambda = float(fake_loss_weight)

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None, **kwargs):
        country_labels = inputs.get("country_labels")
        fake_labels = inputs.get("fake_labels")
        model_inputs = {k: v for k, v in inputs.items()
                        if k not in ("country_labels", "fake_labels", "labels")}
        outputs = model(**model_inputs)
        device = outputs.logits.device
        cw = self._cw.to(device) if self._cw is not None else None
        fw = self._fw.to(device) if self._fw is not None else None
        country_loss = F.cross_entropy(
            outputs.logits, country_labels, weight=cw,
            ignore_index=COUNTRY_IGNORE_INDEX)
        # an all-spoof batch has every country label ignored -> CE returns nan (0/0).
        # In that case it is replaced with 0.
        if torch.isnan(country_loss):
            country_loss = torch.zeros((), device=device)
        fake_loss = F.cross_entropy(outputs.fake_logits, fake_labels, weight=fw)
        loss = country_loss + self._lambda * fake_loss
        return (loss, outputs) if return_outputs else loss


def run_multitask(args) -> None:
    """Joint country + real/fake training entry point (--multitask)."""
    # Fully separate from the country-only path (main): without --multitask the validated
    # country recipe is reproduced unchanged.
    tb_log_dir = os.environ.get(
        "AIP_TENSORBOARD_LOG_DIR", os.path.join(args.output_dir, "tb_logs"))

    # real_fake_5k is already a balanced (real:fake = 35000:35000) flat pool (DATASET.md
    # §11), and gcloud/pad_and_split_v2.py has already computed the speaker-level 70:15:15
    # split column and written it into the manifest — neither per_class/spoof_cap
    # undersampling nor a split recomputation is needed. The split column is simply read here.
    train_df, val_df, test_df = build_multitask_splits(
        real_fake_root=args.real_fake_root,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=SEED,
    )
    report_mt("train", train_df)
    report_mt("val", val_df)
    report_mt("test", test_df)
    manifest_out = os.path.join(args.output_dir, "manifests")
    os.makedirs(manifest_out, exist_ok=True)
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        part.to_csv(os.path.join(manifest_out, f"{name}.csv"), index=False)

    feature_extractor = AutoFeatureExtractor.from_pretrained(args.backbone)
    collator = MultiTaskCollator(feature_extractor)

    train_ds = MultiTaskDataset(train_df, augment=args.augment,
                                aug_strength=args.aug_strength)
    eval_ds = MultiTaskDataset(val_df)
    test_ds = MultiTaskDataset(test_df)
    aug_kind = ("domain" if args.augment and args.aug_strength > 0
                else "legacy" if args.augment else "off")
    print(f"[multitask] train={len(train_ds)} val={len(eval_ds)} test={len(test_ds)} "
          f"augment={args.augment} aug={aug_kind}(strength={args.aug_strength}) "
          f"fake_loss_weight={args.fake_loss_weight}")

    model = AccentClassifier(args.backbone, dropout=args.dropout, head=args.head,
                             layer_weighting=args.layer_weighting, fake_head=True,
                             num_fake_labels=NUM_FAKE_LABELS)
    print(f"backbone={args.backbone} head={args.head} "
          f"layer_weighting={args.layer_weighting} fake_head=True")
    if args.mask_time_prob is not None or args.mask_feature_prob is not None:
        model.set_spec_augment(args.mask_time_prob, args.mask_feature_prob)
    if args.unfreeze_top > 0:
        model.unfreeze_top_layers(args.unfreeze_top)
        print(f"backbone frozen except top {args.unfreeze_top} transformer layers")
    else:
        model.freeze_backbone()
        print("backbone fully frozen (training heads only)")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable:,} / {total:,}")

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
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        # Balanced selection over both heads (mt_macro_f1 = mean(fake, country)). fake saturates
        # quickly on the seen-attack dev set, so the combined metric is used to avoid stopping
        # early before country has matured.
        metric_for_best_model="mt_macro_f1",
        greater_is_better=True,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        # both label tensors are kept + forwarded and returned as a label tuple.
        label_names=["country_labels", "fake_labels"],
        report_to=["tensorboard"],
        logging_dir=tb_log_dir,
        seed=SEED,
    )

    country_weights = compute_country_weights_mt(train_df, args.class_weight)
    fake_weights = compute_fake_weights_mt(train_df, args.class_weight)
    if country_weights is not None:
        print("country weights: %s" % {LABELS[i]: round(float(country_weights[i]), 3)
                                        for i in range(len(LABELS))})
    if fake_weights is not None:
        print("fake weights: %s" % {FAKE_LABELS[i]: round(float(fake_weights[i]), 3)
                                     for i in range(NUM_FAKE_LABELS)})

    callbacks = []
    if args.early_stopping_patience and args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=compute_metrics_multitask,
        country_weights=country_weights,
        fake_weights=fake_weights,
        fake_loss_weight=args.fake_loss_weight,
        callbacks=callbacks,
    )

    trainer.train()
    # Use predict() (instead of evaluate()) to get the scalar metrics together with the
    # raw predictions — as in the country-only path, the raw logits/labels are needed to
    # compute the per-country details (for the model tester) here.
    val_out = trainer.predict(eval_ds, metric_key_prefix="eval")
    val_metrics = val_out.metrics
    print("final val eval:", json.dumps(val_metrics, indent=2))
    test_out = trainer.predict(test_ds, metric_key_prefix="test")
    test_metrics = test_out.metrics
    print("final test eval (held-out, speaker-disjoint; does NOT preserve "
          "ASVspoof's unseen-attack protocol boundary, see DATASET.md §11):",
          json.dumps(test_metrics, indent=2))
    metrics = {**val_metrics, **test_metrics}

    def _country_detail(out) -> dict | None:
        # out.predictions/out.label_ids are (country, fake) tuples (same structure as in
        # compute_metrics_multitask). ASVspoof-derived rows have country_label=-100 and are
        # therefore excluded from the country metrics.
        country_logits, country_labels = out.predictions[0], out.label_ids[0]
        mask = country_labels != COUNTRY_IGNORE_INDEX
        if not mask.any():
            return None
        return detailed_report(np.argmax(country_logits[mask], axis=-1), country_labels[mask])

    # So that the model tester can show multitask jobs the same way as country-only jobs
    # (per-accent F1 bars, confusion matrix), the country head's detailed report is also
    # saved as eval_detail/test_detail (until now only the country_accuracy/country_macro_f1
    # scalars existed, so no per-accent breakdown was possible).
    eval_detail = _country_detail(val_out)
    test_detail = _country_detail(test_out)
    if eval_detail is not None:
        metrics["eval_detail"] = eval_detail
    if test_detail is not None:
        metrics["test_detail"] = test_detail

    metrics["train_config"] = {
        "multitask": True,
        "augment": bool(args.augment),
        "aug_strength": float(args.aug_strength),
        "aug_kind": aug_kind,
        "backbone": args.backbone,
        "head": args.head,
        "unfreeze_top": args.unfreeze_top,
        "real_fake_root": args.real_fake_root,
        "fake_loss_weight": args.fake_loss_weight,
        "epochs": args.epochs,
    }

    trainer.save_model(args.output_dir)
    feature_extractor.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "label_config.json"), "w") as f:
        json.dump({"labels": LABELS, "id2label": ID2LABEL,
                   "fake_labels": FAKE_LABELS, "id2fake": ID2FAKE}, f, indent=2)
    write_model_config(
        args.output_dir, backbone=args.backbone, num_labels=len(LABELS),
        dropout=args.dropout, head=args.head, layer_weighting=args.layer_weighting,
        fake_head=True, num_fake_labels=NUM_FAKE_LABELS)
    with open(os.path.join(args.output_dir, "final_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"saved to {args.output_dir}")


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
                    # Unfreeze the top N transformer layers for training (0 = train the head only)
    ap.add_argument("--gradient-checkpointing", action="store_true")
    # Option to enable gradient checkpointing (trades speed for lower memory use).
    ap.add_argument("--output-dir", default=str(OUTPUT_DIR))
    ap.add_argument("--curated-root", default=str(CURATED_ROOT))
    ap.add_argument("--per-class", type=int, default=None,
                    help="optional speaker-aware cap for quick experiments; "
                         "omit to use the full fixed 5000/class pool (DATASET.md §11)")
    # --- multi-task (country + real/fake) knobs -------------------------------
    ap.add_argument("--multitask", action="store_true",
                    help="joint country + real/fake head training (adds ASVspoof "
                         "spoof corpus; without this flag the country recipe is "
                         "byte-for-byte unchanged)")
    # Joint country + real/fake training (adds the ASVspoof spoof corpus). Without this
    # flag the existing country recipe is reproduced unchanged.
    ap.add_argument("--fake-loss-weight", "--fake_loss_weight", dest="fake_loss_weight",
                    type=float, default=1.0,
                    help="λ multiplier on the real/fake loss (multitask only)")
    # Weight λ of the real/fake loss term in the combined loss (multitask only).
    ap.add_argument("--real-fake-root", "--real_fake_root", dest="real_fake_root",
                    default=str(REAL_FAKE_ROOT),
                    help="root of the pre-balanced curated_spoof/real_fake_5k/ "
                         "pool (multitask only, DATASET.md §11)")
    ap.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    ap.add_argument("--test-fraction", type=float, default=TEST_FRACTION)
    ap.add_argument("--no-fp16", action="store_true")
    # Trains with fp16 (half precision) by default; this flag disables it (e.g. on CPU).
    ap.add_argument("--class-weight", "--class_weight", dest="class_weight",
                    choices=["none", "balanced", "sqrt"], default="balanced",
                    help="per-class loss weighting: none | balanced (1/count) | "
                         "sqrt (tempered) (default: balanced)")
    # Absorb the class imbalance (US/UK/CA ~6k vs CN ~1.17k) through loss weights.
    ap.add_argument("--dropout", type=float, default=0.1,
                    help="classifier head dropout")
    ap.add_argument("--early-stopping-patience", type=int, default=3,
                    help="stop if macro_f1 hasn't improved for N evals (0=disable)")
    # Stop early when macro-F1 has not improved for N evaluations (0 disables it).
    ap.add_argument("--augment", action="store_true",
                    help="light waveform augmentation (gain + noise) on the train split")
    # Apply light waveform augmentation (random gain + Gaussian noise) to the train split
    # only, to improve robustness against the channel confound (GLOBE vs SAA).
    ap.add_argument("--aug-strength", "--aug_strength", dest="aug_strength",
                    type=float, default=0.0,
                    help="with --augment, use domain-randomization augmentation at "
                         "this strength (0=legacy light aug; ~1.0=full). Simulates "
                         "the GLOBE->VoxForge recording-domain shift (speed/band-limit/"
                         "reverb/colored-noise) to fight source-domain overfitting.")
    # Strength of the domain-randomization augmentation when used with --augment. 0 = legacy
    # light augmentation, ~1.0 = full strength. Mimics the GLOBE→VoxForge recording-domain
    # shift (speed, band-limiting, reverb, colored noise) to reduce overfitting to the source
    # domain (countermeasure C, targeting the domain-shift half of the CA→US collapse).
    ap.add_argument("--save-total-limit", "--save_total_limit", dest="save_total_limit",
                    type=int, default=2,
                    help="max checkpoints to keep (default 2). Raise to keep every "
                         "epoch for a checkpoint scan (e.g. OOD-optimal early-stopping).")
    # Maximum number of checkpoints to keep (default 2). Raise it (e.g. to at least the
    # number of epochs) to keep every epoch's checkpoint and scan for the best OOD
    # (VoxForge) stopping point.
    ap.add_argument("--hypertune", action="store_true",
                    help="report eval_macro_f1 to Vertex AI Vizier (HP tuning jobs)")
    # Enable only when reporting trial scores from a Vertex AI Hyperparameter Tuning (Vizier) job.
    # --- architecture knobs (v3 experiments) ---------------------------------
    ap.add_argument("--backbone", "--model-name", "--model_name", dest="backbone",
                    default=MODEL_NAME,
                    help="pretrained backbone (e.g. facebook/wav2vec2-base or "
                         "microsoft/wavlm-base-plus)")
    # For swapping the backbone. AutoModel picks wav2vec2/wavlm automatically from the name.
    ap.add_argument("--head", "--head_type", dest="head",
                    choices=["mean", "attentive"], default="mean",
                    help="utterance pooling head: mean (masked mean) | attentive "
                         "(attentive statistics pooling: weighted mean+std)")
    # Utterance pooling head: mean (masked mean) | attentive (attention-weighted mean + std).
    ap.add_argument("--layer-weighting", "--layer_weighting", dest="layer_weighting",
                    action="store_true",
                    help="learned weighted sum over all backbone layers (SUPERB-style)")
    # Use a learnable weighted sum of the hidden states of all backbone layers as the
    # representation (SUPERB-style).
    ap.add_argument("--mask-time-prob", "--mask_time_prob", dest="mask_time_prob",
                    type=float, default=None,
                    help="SpecAugment time-mask prob (None=backbone default ~0.05)")
    ap.add_argument("--mask-feature-prob", "--mask_feature_prob", dest="mask_feature_prob",
                    type=float, default=None,
                    help="SpecAugment feature-mask prob (None=backbone default)")
    # Strength of the backbone's built-in SpecAugment (time/channel feature masking during
    # training). None = backbone default.
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Multitask (country + real/fake) branches into its own path. When this branch is not
    # taken, the country-only code below stays exactly as before (guards against recipe
    # regressions).
    if args.multitask:
        run_multitask(args)
        return

    # Vertex AI Custom Jobs configured with a TensorBoard resource + service
    # account inject AIP_TENSORBOARD_LOG_DIR (a GCS path) and continuously sync
    # anything written there to the TensorBoard instance while the job runs.
    # Falls back to a local dir for plain (non-Vertex) runs.
    tb_log_dir = os.environ.get(
        "AIP_TENSORBOARD_LOG_DIR", os.path.join(args.output_dir, "tb_logs")
    )

    # Build the speaker-level train/val/test splits on the fly from the curated manifests.
    # The original curated/ is only read; the split CSVs are written under output_dir only.
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

    # Prepare the input preprocessor (handles normalization) and the batch collator that uses
    # it. AutoFeatureExtractor loads the matching preprocessor for either backbone (wav2vec2/wavlm).
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
        # Adjust the strength of the backbone's built-in SpecAugment (applied in training only).
        model.set_spec_augment(args.mask_time_prob, args.mask_feature_prob)
        print(f"spec-augment: mask_time_prob={args.mask_time_prob} "
              f"mask_feature_prob={args.mask_feature_prob}")
    if args.unfreeze_top > 0:
        # Mode that also fine-tunes the top N layers of the backbone.
        model.unfreeze_top_layers(args.unfreeze_top)
        print(f"backbone frozen except top {args.unfreeze_top} transformer layers")
    else:
        # Default mode: backbone fully frozen, head only (the fastest and safest starting point).
        model.freeze_backbone()
        print("backbone fully frozen (training head only)")
    # Log the trainable vs. total parameter counts as a sanity check.
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable:,} / {total:,}")

    # --- diagnostic: do pooled logits actually vary across different clips? ---
    # Check that different clips yield different pooled outputs. A batch-std close to 0 means
    # the features have collapsed (every input maps to practically the same representation);
    # the head then cannot separate the classes and the loss stays stuck at ln(num_classes).
    # Printed once, before training starts.
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
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch",   # evaluate every epoch
        save_strategy="epoch",   # save a checkpoint every epoch
        logging_steps=50,
        load_best_model_at_end=True,       # load the best checkpoint when training ends
        metric_for_best_model="macro_f1",  # "best" is judged by macro-F1
        greater_is_better=True,
        save_total_limit=args.save_total_limit,   # default 2 (saves disk); raise it when scanning
        dataloader_num_workers=4,
        remove_unused_columns=False,  # our model consumes raw batch dict
        # By default the HF Trainer drops batch columns that are not in the model's forward
        # signature, but our model consumes the batch dict built by the collator as-is,
        # so this automatic removal has to be turned off.
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
    # Use predict() (instead of evaluate()) to get the scalar metrics and the raw predictions.
    # The raw predictions are required for the confusion matrix and per-class precision/recall.
    val_out = trainer.predict(eval_ds, metric_key_prefix="eval")
    val_metrics = val_out.metrics
    print("final val eval:", json.dumps(val_metrics, indent=2))
    # Report this trial's score to Vertex AI Hyperparameter Tuning (Vizier).
    # Runs only with --hypertune and only when cloudml-hypertune is installed.
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
    # Also measure the final performance on the held-out test set never used in training.
    test_out = trainer.predict(test_ds, metric_key_prefix="test")
    test_metrics = test_out.metrics
    print("final test eval:", json.dumps(test_metrics, indent=2))
    metrics = {**val_metrics, **test_metrics}
    # Record this job's augmentation settings for reproducibility and tracking (extra keys,
    # so they are safe for the dashboard schema).
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
    # Also store the per-country detailed metrics + confusion matrix (val and test) under nested keys.
    metrics["eval_detail"] = detailed_report(
        np.argmax(val_out.predictions, axis=-1), val_out.label_ids)
    metrics["test_detail"] = detailed_report(
        np.argmax(test_out.predictions, axis=-1), test_out.label_ids)

    # persist head + backbone weights, feature extractor, and label config
    # to output_dir so that evaluate.py / infer.py can load them later as-is.
    trainer.save_model(args.output_dir)
    feature_extractor.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "label_config.json"), "w") as f:
        json.dump({"labels": LABELS, "id2label": ID2LABEL}, f, indent=2)
    # Also record the architecture hyperparameters so that the inference side
    # (infer/evaluate/model_tester) can rebuild the model with the same structure as the
    # saved weights.
    write_model_config(
        args.output_dir, backbone=args.backbone, num_labels=len(LABELS),
        dropout=args.dropout, head=args.head, layer_weighting=args.layer_weighting)
    with open(os.path.join(args.output_dir, "final_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"saved to {args.output_dir}")


if __name__ == "__main__":
    main()
