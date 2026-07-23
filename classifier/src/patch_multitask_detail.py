"""One-off patch: backfill per-country ``eval_detail``/``test_detail`` into an
already-trained multitask model's ``final_metrics.json``.

Why this exists: the multitask training path (``train.py --multitask``)
originally only wrote aggregate ``country_accuracy``/``country_macro_f1``
scalars, not the per-class precision/recall/f1/support + confusion-matrix
block the model tester dashboard needs to show an accent-by-accent breakdown
(it already showed real/fake fine, since those come from flat
``test_fake_f1_<label>`` scalars). ``train.py`` now computes this for future
runs; this script re-derives it for models trained before that fix, without
re-training. Reuses ``Trainer.predict()`` (the same code path that
successfully scored test.csv *during* training) rather than a hand-rolled
DataLoader loop — a standalone ``evaluate.py`` CLI eval hung repeatedly on
both CPU and GPU jobs for reasons that were never root-caused; this sidesteps
that path entirely.

Usage:
    python patch_multitask_detail.py --model-dir /gcs/<bucket>/outputs/classifier/<job>/model \
        --extra-copy /gcs/<mirror-bucket>/outputs/classifier/<job>/model
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np
from transformers import AutoFeatureExtractor, TrainingArguments

from config import COUNTRY_IGNORE_INDEX
from dataset import MultiTaskCollator, MultiTaskDataset
from evaluate import load_trained
from train import MultiTaskTrainer, detailed_report


def _country_detail(out) -> dict | None:
    country_logits, country_labels = out.predictions[0], out.label_ids[0]
    mask = country_labels != COUNTRY_IGNORE_INDEX
    if not mask.any():
        return None
    return detailed_report(np.argmax(country_logits[mask], axis=-1), country_labels[mask])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--extra-copy", default=None,
                    help="also write the patched final_metrics.json here (e.g. the "
                         "model-tester mirror bucket)")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    model = load_trained(args.model_dir)
    fe = AutoFeatureExtractor.from_pretrained(
        json.loads(open(os.path.join(args.model_dir, "model_config.json")).read())
        .get("backbone", "microsoft/wavlm-base-plus"))
    collator = MultiTaskCollator(fe)

    manifest_dir = os.path.join(args.model_dir, "manifests")
    val_ds = MultiTaskDataset(os.path.join(manifest_dir, "val.csv"))
    test_ds = MultiTaskDataset(os.path.join(manifest_dir, "test.csv"))

    training_args = TrainingArguments(
        output_dir="/tmp/patch_detail",
        per_device_eval_batch_size=args.batch_size,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        label_names=["country_labels", "fake_labels"],
        report_to=[],
    )
    trainer = MultiTaskTrainer(model=model, args=training_args, data_collator=collator)

    val_out = trainer.predict(val_ds, metric_key_prefix="eval")
    test_out = trainer.predict(test_ds, metric_key_prefix="test")

    metrics_path = os.path.join(args.model_dir, "final_metrics.json")
    metrics = json.loads(open(metrics_path).read())

    eval_detail = _country_detail(val_out)
    test_detail = _country_detail(test_out)
    if eval_detail is not None:
        metrics["eval_detail"] = eval_detail
    if test_detail is not None:
        metrics["test_detail"] = test_detail

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"patched {metrics_path}")

    if args.extra_copy:
        os.makedirs(args.extra_copy, exist_ok=True)
        dest = os.path.join(args.extra_copy, "final_metrics.json")
        shutil.copyfile(metrics_path, dest)
        print(f"also wrote {dest}")


if __name__ == "__main__":
    main()
