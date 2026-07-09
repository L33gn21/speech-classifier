"""Gradio web UI to test the trained accent classifier.

Record from the mic or upload a clip; get per-accent proximity percentages
(Level 1) plus a frame-level probability heatmap over time (Level 2).

Run:
    .venv/bin/python src/classifier/webui.py
    .venv/bin/python src/classifier/webui.py --model-dir outputs/classifier --share

Then open the printed http://127.0.0.1:7860 URL.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import OUTPUT_DIR, SAMPLE_RATE
from infer import load_trained, predict

# Populated once at startup by main().
MODEL = None
FEATURE_EXTRACTOR = None
LABELS: list[str] = []

# Stable, colorblind-friendly-ish colors per accent (order = LABELS).
_PALETTE = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2"]


def _color(i: int) -> str:
    return _PALETTE[i % len(_PALETTE)]


def _heatmap_figure(frame_probs: np.ndarray, labels: list[str]):
    """Line plot of per-accent probability across time (~20ms per frame)."""
    t = np.arange(frame_probs.shape[0]) * 0.02
    fig, ax = plt.subplots(figsize=(9, 3.2))
    for c, name in enumerate(labels):
        ax.plot(t, frame_probs[:, c], label=name, color=_color(c), linewidth=1.8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("probability")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(0, max(t[-1], 0.1) if len(t) else 0.1)
    ax.set_title("Frame-level accent probability over time (Level 2)")
    ax.legend(loc="upper right", ncols=len(labels), fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def classify(audio_path: str | None):
    """Gradio callback. Returns (label->prob dict, summary md, heatmap figure)."""
    if not audio_path:
        return {}, "🎤 오디오를 녹음하거나 업로드해줘.", None

    probs, frame_probs = predict(MODEL, FEATURE_EXTRACTOR, Path(audio_path), want_frames=True)

    # gr.Label expects {label: confidence}; it sorts/bars automatically.
    conf = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}

    order = np.argsort(probs)[::-1]
    top = LABELS[order[0]]
    lines = [f"**추정 억양: `{top}` ({probs[order[0]] * 100:.1f}%)**", "", "| 억양 | 근접도 |", "|---|---|"]
    for i in order:
        lines.append(f"| {LABELS[i]} | {probs[i] * 100:.1f}% |")
    summary = "\n".join(lines)

    fig = _heatmap_figure(frame_probs, LABELS)
    return conf, summary, fig


def build_demo():
    import gradio as gr

    with gr.Blocks(title="Accent Classifier") as demo:
        gr.Markdown(
            "# 🗣️ 영어 억양 분류기\n"
            f"발화를 입력하면 **{', '.join(LABELS)}** 억양에 대한 근접도를 퍼센트로 보여줘.\n"
            "아래에서 마이크로 녹음하거나 오디오 파일을 올리고 **분류** 버튼을 눌러."
        )
        with gr.Row():
            with gr.Column(scale=1):
                audio_in = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="발화 입력 (녹음 / 업로드)",
                )
                run_btn = gr.Button("분류", variant="primary")
            with gr.Column(scale=1):
                label_out = gr.Label(label="억양 근접도", num_top_classes=len(LABELS))
                summary_out = gr.Markdown()
        heatmap_out = gr.Plot(label="시간축 억양 확률 (Level 2)")

        run_btn.click(classify, inputs=audio_in, outputs=[label_out, summary_out, heatmap_out])
        # also auto-run when a clip is recorded/uploaded
        audio_in.change(classify, inputs=audio_in, outputs=[label_out, summary_out, heatmap_out])

    return demo


def main() -> None:
    global MODEL, FEATURE_EXTRACTOR, LABELS

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(OUTPUT_DIR))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="expose a public gradio.live link")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading model from {args.model_dir} onto {device} ...")
    MODEL, FEATURE_EXTRACTOR, LABELS = load_trained(Path(args.model_dir))
    MODEL.to(device)
    print(f"labels: {LABELS}")

    demo = build_demo()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
