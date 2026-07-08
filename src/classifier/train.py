"""Fine-tune the accent classifier with HuggingFace Trainer.

Default recipe: freeze the whole wav2vec2 backbone and train only the linear
head. Pass --unfreeze-top N to also fine-tune the top N transformer layers
(do this once the head alone is working).

Example:
    python src/train.py --epochs 8 --batch-size 8 --grad-accum 2
    python src/train.py --unfreeze-top 4 --lr 2e-5 --epochs 6
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import Trainer, TrainingArguments, Wav2Vec2FeatureExtractor

from config import ID2LABEL, LABELS, MANIFEST_DIR, MODEL_NAME, OUTPUT_DIR, SEED
from dataset import AccentDataset, DataCollator
from model import AccentClassifier


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    per_class_f1 = f1_score(labels, preds, average=None, labels=list(range(len(LABELS))))
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }
    for i, name in ID2LABEL.items():
        metrics[f"f1_{name}"] = float(per_class_f1[i])
    return metrics


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
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--output-dir", default=str(OUTPUT_DIR))
    ap.add_argument("--no-fp16", action="store_true")
    args = ap.parse_args()

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    collator = DataCollator(feature_extractor)

    train_ds = AccentDataset(MANIFEST_DIR / "train.csv")
    eval_ds = AccentDataset(MANIFEST_DIR / "test.csv")
    print(f"train={len(train_ds)}  eval={len(eval_ds)}")

    model = AccentClassifier(MODEL_NAME)
    if args.unfreeze_top > 0:
        model.unfreeze_top_layers(args.unfreeze_top)
        print(f"backbone frozen except top {args.unfreeze_top} transformer layers")
    else:
        model.freeze_backbone()
        print("backbone fully frozen (training head only)")
    # gradient checkpointing is enabled by the Trainer via TrainingArguments
    # (which calls model.gradient_checkpointing_enable) — no manual call here.
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
        # use_reentrant=False so checkpointing works when lower backbone layers
        # are frozen (reentrant mode errors when no checkpoint input needs grad).
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        dataloader_num_workers=4,
        remove_unused_columns=False,  # our model consumes raw batch dict
        report_to="none",
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
    metrics = trainer.evaluate()
    print("final eval:", json.dumps(metrics, indent=2))

    # persist head + backbone weights, feature extractor, and label config
    trainer.save_model(args.output_dir)
    feature_extractor.save_pretrained(args.output_dir)
    with open(f"{args.output_dir}/label_config.json", "w") as f:
        json.dump({"labels": LABELS, "id2label": ID2LABEL}, f, indent=2)
    print(f"saved to {args.output_dir}")


if __name__ == "__main__":
    main()
