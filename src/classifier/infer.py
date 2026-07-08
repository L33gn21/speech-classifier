"""Run a trained accent classifier on an audio file.

Outputs per-accent proximity percentages (Level 1). With --frames it also
returns frame-level probabilities (Level 2 time-axis heatmap material).

Example:
    python src/infer.py path/to/clip.mp3
    python src/infer.py clip.mp3 --frames --plot heatmap.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor

from config import LABELS, MODEL_NAME, OUTPUT_DIR, SAMPLE_RATE
from dataset import load_audio
from model import AccentClassifier


def load_trained(model_dir: Path) -> tuple[AccentClassifier, Wav2Vec2FeatureExtractor, list[str]]:
    model = AccentClassifier(MODEL_NAME)
    state = None
    safepath = model_dir / "model.safetensors"
    binpath = model_dir / "pytorch_model.bin"
    if safepath.exists():
        from safetensors.torch import load_file

        state = load_file(str(safepath))
    elif binpath.exists():
        state = torch.load(str(binpath), map_location="cpu")
    else:
        raise FileNotFoundError(f"no weights (model.safetensors / pytorch_model.bin) in {model_dir}")
    model.load_state_dict(state)
    model.eval()

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir)
    labels = LABELS
    cfg = model_dir / "label_config.json"
    if cfg.exists():
        labels = json.loads(cfg.read_text())["labels"]
    return model, feature_extractor, labels


@torch.no_grad()
def predict(model, feature_extractor, audio_path: Path, want_frames: bool = False):
    wav = load_audio(audio_path)
    inputs = feature_extractor(
        [wav], sampling_rate=SAMPLE_RATE, return_attention_mask=True, return_tensors="pt"
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model(**inputs, output_frame_logits=want_frames)
    probs = torch.softmax(out.logits, dim=-1)[0].cpu().numpy()
    frame_probs = None
    if want_frames:
        frame_probs = torch.softmax(out.frame_logits, dim=-1)[0].cpu().numpy()  # [T, C]
    return probs, frame_probs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--model-dir", default=str(OUTPUT_DIR))
    ap.add_argument("--frames", action="store_true", help="also compute frame-level probs")
    ap.add_argument("--plot", default=None, help="save a frame-level heatmap PNG (implies --frames)")
    args = ap.parse_args()

    model, fe, labels = load_trained(Path(args.model_dir))
    want_frames = args.frames or args.plot is not None
    probs, frame_probs = predict(model, fe, Path(args.audio), want_frames)

    order = np.argsort(probs)[::-1]
    print(f"\nAccent proximity for {args.audio}:")
    for i in order:
        print(f"  {labels[i]:10s} {probs[i] * 100:5.1f}%")

    if args.plot is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 3))
        # ~20ms per wav2vec2 frame; x in seconds
        t = np.arange(frame_probs.shape[0]) * 0.02
        for c, name in enumerate(labels):
            ax.plot(t, frame_probs[:, c], label=name)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("prob")
        ax.set_title("frame-level accent probabilities")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        print(f"saved heatmap to {args.plot}")


if __name__ == "__main__":
    main()
