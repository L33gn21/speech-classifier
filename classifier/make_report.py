#!/usr/bin/env python
"""Evaluate a trained accent classifier and render presentation-quality figures.

Runs after train.py on the same box (GPU). Loads the saved model + the
speaker-disjoint test manifest, runs inference (predictions + pooled wav2vec2
embeddings in one pass), and writes to <out>/:

  metrics.json               accuracy, macro-F1, per-class P/R/F1, confusion matrix
  confusion_matrix.png       counts (rows=true, cols=pred)
  confusion_matrix_norm.png  row-normalized (recall) heatmap
  per_class_f1.png           precision / recall / F1 bars per class
  training_curves.png        train loss + eval accuracy/macro-F1 over epochs
  dataset_composition.png    clips per class (train/val/test) + gender + source
  embeddings_pca.png         2-D PCA of test embeddings, coloured by true class

Every figure is wrapped in try/except so one failure never sinks the rest.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import CURATED_ROOT, LABELS, MODEL_NAME, OUTPUT_DIR
from dataset import AccentDataset, DataCollator
from model import AccentClassifier
from transformers import Wav2Vec2FeatureExtractor

plt.rcParams.update({"figure.dpi": 150, "font.size": 11, "axes.grid": False,
                     "figure.autolayout": True})
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860",
           "#DA8BC3", "#8C8C8C"]


def log(*a):
    print(*a, flush=True)


def load_trained(model_dir: str) -> AccentClassifier:
    model = AccentClassifier(MODEL_NAME, pretrained=False)
    safe = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(safe):
        from safetensors.torch import load_file
        state = load_file(safe)
    else:
        state = torch.load(os.path.join(model_dir, "pytorch_model.bin"), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def infer(model, ds, collator, device, want_emb=True):
    loader = DataLoader(ds, batch_size=16, collate_fn=collator, num_workers=2)
    preds, labels, embs = [], [], []
    for batch in loader:
        y = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model.wav2vec2(batch["input_values"], attention_mask=batch.get("attention_mask"))
        hidden = out.last_hidden_state
        am = batch.get("attention_mask")
        if am is not None:
            fm = model.wav2vec2._get_feature_vector_attention_mask(hidden.shape[1], am).unsqueeze(-1)
            emb = (hidden * fm).sum(1) / fm.sum(1).clamp(min=1)
        else:
            emb = hidden.mean(1)
        logits = model.classifier(emb)
        preds.append(logits.argmax(-1).cpu().numpy())
        labels.append(y.numpy())
        if want_emb:
            embs.append(emb.cpu().numpy())
    return (np.concatenate(preds), np.concatenate(labels),
            np.concatenate(embs) if want_emb else None)


def fig_confusion(cm, labels, out, normalize=False):
    m = cm.astype(float)
    title = "Confusion matrix (counts)"
    fmt = lambda v: f"{int(v)}"
    if normalize:
        m = m / m.sum(1, keepdims=True).clip(min=1)
        title = "Confusion matrix (row-normalized = recall)"
        fmt = lambda v: f"{v:.2f}"
    fig, ax = plt.subplots(figsize=(1.3 * len(labels) + 2, 1.3 * len(labels) + 1.5))
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=m.max() if not normalize else 1.0)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    thr = m.max() * 0.6 if not normalize else 0.6
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, fmt(m[i, j]), ha="center", va="center",
                    color="white" if m[i, j] > thr else "black", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(out); plt.close(fig)


def fig_per_class(labels, p, r, f, out):
    x = np.arange(len(labels)); w = 0.26
    fig, ax = plt.subplots(figsize=(1.4 * len(labels) + 2, 4.5))
    ax.bar(x - w, p, w, label="precision", color="#4C72B0")
    ax.bar(x, r, w, label="recall", color="#55A868")
    ax.bar(x + w, f, w, label="F1", color="#DD8452")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05); ax.set_ylabel("score")
    ax.set_title("Per-class precision / recall / F1 (held-out test)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(f):
        ax.text(i + w, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    fig.savefig(out); plt.close(fig)


def fig_training_curves(model_dir, out):
    states = glob.glob(os.path.join(model_dir, "**", "trainer_state.json"), recursive=True)
    if not states:
        log("  no trainer_state.json — skipping curves"); return
    best = max(states, key=lambda p: len(json.load(open(p)).get("log_history", [])))
    hist = json.load(open(best))["log_history"]
    tr = [(h["step"], h["loss"]) for h in hist if "loss" in h and "eval_loss" not in h]
    ev = [(h["epoch"], h.get("eval_accuracy"), h.get("eval_macro_f1")) for h in hist if "eval_accuracy" in h]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    if tr:
        s, l = zip(*tr); a1.plot(s, l, color="#C44E52")
    a1.set_xlabel("step"); a1.set_ylabel("train loss"); a1.set_title("Training loss")
    a1.grid(alpha=0.3)
    if ev:
        e, acc, mf1 = zip(*ev)
        a2.plot(e, acc, "-o", label="accuracy", color="#4C72B0")
        a2.plot(e, mf1, "-o", label="macro-F1", color="#55A868")
        a2.set_ylim(0, 1.02); a2.legend()
    a2.set_xlabel("epoch"); a2.set_ylabel("score"); a2.set_title("Validation accuracy / macro-F1")
    a2.grid(alpha=0.3)
    fig.savefig(out); plt.close(fig)


def fig_dataset(model_dir, curated_root, out):
    mdir = os.path.join(model_dir, "manifests")
    splits = {}
    for s in ("train", "val", "test"):
        p = os.path.join(mdir, f"{s}.csv")
        if os.path.exists(p):
            splits[s] = pd.read_csv(p)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    # 1) clips per class stacked by split
    bottom = np.zeros(len(LABELS))
    colors = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}
    for s, df in splits.items():
        counts = df["country"].value_counts().reindex(LABELS, fill_value=0).values
        axes[0].bar(LABELS, counts, bottom=bottom, label=s, color=colors.get(s))
        bottom += counts
    axes[0].set_title("Clips per class (by split)"); axes[0].set_ylabel("clips")
    axes[0].legend(); axes[0].grid(axis="y", alpha=0.3)
    # 2) gender balance per class (from curated manifests)
    gf, gm, gu = [], [], []
    for cc in LABELS:
        mp = os.path.join(curated_root, cc, "manifest.csv")
        if os.path.exists(mp):
            m = pd.read_csv(mp, dtype=str, keep_default_na=False)
            gf.append((m["gender"] == "F").sum()); gm.append((m["gender"] == "M").sum())
            gu.append((~m["gender"].isin(["F", "M"])).sum())
        else:
            gf.append(0); gm.append(0); gu.append(0)
    gf, gm, gu = np.array(gf), np.array(gm), np.array(gu)
    axes[1].bar(LABELS, gf, label="F", color="#DA8BC3")
    axes[1].bar(LABELS, gm, bottom=gf, label="M", color="#4C72B0")
    axes[1].bar(LABELS, gu, bottom=gf + gm, label="U", color="#8C8C8C")
    axes[1].set_title("Gender balance per class (curated pool)"); axes[1].set_ylabel("clips")
    axes[1].legend(); axes[1].grid(axis="y", alpha=0.3)
    # 3) source breakdown per class
    src_names, src_counts = set(), {}
    for cc in LABELS:
        mp = os.path.join(curated_root, cc, "manifest.csv")
        if os.path.exists(mp):
            m = pd.read_csv(mp, dtype=str, keep_default_na=False)
            vc = m["source"].value_counts().to_dict()
            src_counts[cc] = vc; src_names |= set(vc)
    src_names = sorted(src_names)
    bottom = np.zeros(len(LABELS))
    for k, sname in enumerate(src_names):
        vals = np.array([src_counts.get(cc, {}).get(sname, 0) for cc in LABELS])
        axes[2].bar(LABELS, vals, bottom=bottom, label=sname, color=PALETTE[k % len(PALETTE)])
        bottom += vals
    axes[2].set_title("Source breakdown per class (curated pool)"); axes[2].set_ylabel("clips")
    axes[2].legend(); axes[2].grid(axis="y", alpha=0.3)
    fig.savefig(out); plt.close(fig)


def fig_pca(emb, labels, out):
    if emb is None or len(emb) < 3:
        return
    xy = PCA(n_components=2, random_state=42).fit_transform(emb)
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, cc in enumerate(LABELS):
        m = labels == i
        if m.any():
            ax.scatter(xy[m, 0], xy[m, 1], s=12, alpha=0.6, color=PALETTE[i % len(PALETTE)], label=cc)
    ax.set_title("wav2vec2 pooled embeddings — PCA (test set, colour = true class)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(markerscale=2)
    fig.savefig(out); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(OUTPUT_DIR))
    ap.add_argument("--curated-root", default=str(CURATED_ROOT))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.model_dir, "figures")
    os.makedirs(out, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_trained(args.model_dir).to(device)
    fe = Wav2Vec2FeatureExtractor.from_pretrained(args.model_dir)
    collator = DataCollator(fe)

    test_csv = os.path.join(args.model_dir, "manifests", "test.csv")
    test_ds = AccentDataset(test_csv, curated_root=args.curated_root)
    log(f"inference on {len(test_ds)} test clips ({device})")
    preds, labels, emb = infer(model, test_ds, collator, device)

    ids = list(range(len(LABELS)))
    acc = float(accuracy_score(labels, preds))
    mf1 = float(f1_score(labels, preds, average="macro", labels=ids))
    p, r, f, sup = precision_recall_fscore_support(labels, preds, labels=ids, zero_division=0)
    cm = confusion_matrix(labels, preds, labels=ids)
    metrics = {"accuracy": acc, "macro_f1": mf1, "labels": LABELS,
               "per_class": {LABELS[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                          "f1": float(f[i]), "support": int(sup[i])} for i in ids},
               "confusion_matrix": cm.tolist(), "n_test": int(len(labels))}
    json.dump(metrics, open(os.path.join(out, "metrics.json"), "w"), indent=2)
    log(f"accuracy={acc:.4f} macro_f1={mf1:.4f}")

    for name, fn in [
        ("confusion_matrix.png", lambda o: fig_confusion(cm, LABELS, o, False)),
        ("confusion_matrix_norm.png", lambda o: fig_confusion(cm, LABELS, o, True)),
        ("per_class_f1.png", lambda o: fig_per_class(LABELS, p, r, f, o)),
        ("training_curves.png", lambda o: fig_training_curves(args.model_dir, o)),
        ("dataset_composition.png", lambda o: fig_dataset(args.model_dir, args.curated_root, o)),
        ("embeddings_pca.png", lambda o: fig_pca(emb, labels, o)),
    ]:
        try:
            fn(os.path.join(out, name)); log("  wrote", name)
        except Exception as e:
            log(f"  !! {name} failed: {e!r}")
    log("report done ->", out)


if __name__ == "__main__":
    main()
