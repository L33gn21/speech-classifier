"""Run a trained accent classifier on an audio file.

Outputs per-accent proximity percentages (Level 1). With --frames it also
returns frame-level probabilities (Level 2 time-axis heatmap material).

Example:
    python infer.py path/to/clip.mp3
    python infer.py clip.mp3 --frames --plot heatmap.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoFeatureExtractor

from config import FAKE_LABELS, LABELS, OUTPUT_DIR, SAMPLE_RATE
from dataset import load_audio
from model import AccentClassifier, load_from_dir


def load_trained(model_dir: Path) -> tuple[AccentClassifier, "AutoFeatureExtractor", list[str]]:
    # Load the model weights, feature extractor and label list from the saved model
    # directory. If label_config.json exists, its label order takes precedence (in case it
    # differs from config.LABELS at training time).
    # load_from_dir reads model_config.json and builds the skeleton with the same backbone,
    # head and layer-weighting structure as in training (older checkpoints fall back to the
    # legacy defaults).
    model = load_from_dir(model_dir)
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

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_dir)
    labels = LABELS
    cfg = model_dir / "label_config.json"
    if cfg.exists():
        labels = json.loads(cfg.read_text())["labels"]
    return model, feature_extractor, labels


@torch.no_grad()
def predict(model, feature_extractor, audio_path: Path, want_frames: bool = False):
    # Load and preprocess a single audio file, run it through the model, and return the
    # utterance-level probabilities (plus per-frame probabilities on request, and real/fake
    # probabilities when the model has a fake head).
    wav = load_audio(audio_path)
    inputs = feature_extractor(
        [wav], sampling_rate=SAMPLE_RATE, return_attention_mask=True, return_tensors="pt"
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model(**inputs, output_frame_logits=want_frames)
    # softmax turns the logits into a probability distribution. The batch size is 1, so take [0].
    probs = torch.softmax(out.logits, dim=-1)[0].cpu().numpy()
    frame_probs = None
    # The attentive pooling head provides no frame logits (the utterance logits are not a
    # plain mean of the frames). frame_probs stays None in that case.
    if want_frames and out.frame_logits is not None:
        frame_probs = torch.softmax(out.frame_logits, dim=-1)[0].cpu().numpy()  # [T, C]
    # For models with a real/fake head, also return the binary probabilities (None otherwise).
    fake_probs = None
    if getattr(model, "fake_head_enabled", False) and out.fake_logits is not None:
        fake_probs = torch.softmax(out.fake_logits, dim=-1)[0].cpu().numpy()  # [2]
    return probs, frame_probs, fake_probs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--model-dir", default=str(OUTPUT_DIR))
    ap.add_argument("--frames", action="store_true", help="also compute frame-level probs")
    # Whether to also compute frame-level probabilities.
    ap.add_argument("--plot", default=None, help="save a frame-level heatmap PNG (implies --frames)")
    # Path for the heatmap PNG. When given, --frames is treated as enabled automatically.
    args = ap.parse_args()

    model, fe, labels = load_trained(Path(args.model_dir))
    want_frames = args.frames or args.plot is not None
    probs, frame_probs, fake_probs = predict(model, fe, Path(args.audio), want_frames)

    # For models with a real/fake head, print the spoof verdict first (the most important one).
    if fake_probs is not None:
        fake_labels = FAKE_LABELS
        cfg = Path(args.model_dir) / "label_config.json"
        if cfg.exists():
            fake_labels = json.loads(cfg.read_text()).get("fake_labels", FAKE_LABELS)
        verdict = fake_labels[int(np.argmax(fake_probs))]
        print(f"\nReal/Fake verdict for {args.audio}: {verdict.upper()}")
        for i, name in enumerate(fake_labels):
            print(f"  {name:10s} {fake_probs[i] * 100:5.1f}%")

    # Print sorted by descending probability (the most likely accent first).
    order = np.argsort(probs)[::-1]
    print(f"\nAccent proximity for {args.audio}:")
    for i in order:
        print(f"  {labels[i]:10s} {probs[i] * 100:5.1f}%")

    if args.plot is not None and frame_probs is None:
        print("(no frame-level heatmap: this model uses the attentive pooling head)")
    elif args.plot is not None:
        import matplotlib

        matplotlib.use("Agg")  # fixed backend so it also works without a GUI (e.g. on servers)
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
