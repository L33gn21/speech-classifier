"""Unified entry point of the speech pipeline (detector -> classifier).

Takes one utterance and runs a two-stage pipeline.
  1) detector : decides AI-synthesized (fake) vs. real human (real)
  2) classifier: estimates the proximity (%) to English accent regions, only when judged real

detector and classifier each use flat imports such as `from model import ...` /
`from config import ...`, and both directories contain model.py/config.py/dataset.py.
Putting both packages on sys.path at the same time therefore makes the names collide. The
`_import_context` context manager below exposes only one package on the import path at a
time and clears the colliding module cache, so each subsystem is loaded independently.

Python API:
    from app import SpeechPipeline
    pipe = SpeechPipeline()
    print(pipe.analyze("clip.wav"))

CLI (single-file analysis):
    .venv/bin/python src/app.py path/to/clip.wav [--frames] [--json]

Web demo (microphone/upload -> detector verdict + classifier accent % + frame heatmap):
    .venv/bin/python src/app.py [--host 127.0.0.1] [--port 7860] [--share]
"""
from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DETECTOR_DIR = PROJECT_ROOT / "src" / "detector"
CLASSIFIER_DIR = PROJECT_ROOT / "src" / "classifier"

DETECTOR_WEIGHTS = PROJECT_ROOT / "outputs" / "detector" / "detector.pt"
CLASSIFIER_DIR_OUT = PROJECT_ROOT / "outputs" / "classifier"

# Flat module names shared by the two subpackages — cleared from the cache before importing.
_CONFLICTING_MODULES = ("config", "model", "dataset", "infer", "inference")


@contextlib.contextmanager
def _import_context(pkg_dir: Path):
    """Import with only `pkg_dir` at the front of sys.path and the colliding module cache cleared."""
    saved_path = list(sys.path)
    saved_modules = {name: sys.modules.pop(name) for name in _CONFLICTING_MODULES if name in sys.modules}
    sys.path.insert(0, str(pkg_dir))
    try:
        yield
    finally:
        sys.path[:] = saved_path
        # Remove the flat modules newly loaded in this context and restore the ones that existed before.
        for name in _CONFLICTING_MODULES:
            sys.modules.pop(name, None)
        sys.modules.update(saved_modules)


class DetectorModel:
    """Stage 1: real/fake decision with log-mel + resnet18."""

    def __init__(self, weights: Path = DETECTOR_WEIGHTS, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        with _import_context(DETECTOR_DIR):
            from model import Detector  # src/detector/model.py

            model = Detector().to(self.device)
            model.load_state_dict(torch.load(str(weights), map_location=self.device))
            model.eval()
        self.model = model

    @staticmethod
    def _to_logmel(audio_path: Path) -> np.ndarray:
        """Same preprocessing as inference.py: 16 kHz -> 128-mel log, crop/pad to 128 frames, standardize."""
        import librosa

        audio, sr = librosa.load(str(audio_path), sr=16000)
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        mel = librosa.power_to_db(mel)
        if mel.shape[1] < 128:
            mel = np.pad(mel, ((0, 0), (0, 128 - mel.shape[1])))
        mel = mel[:, :128]
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        return mel

    @torch.no_grad()
    def predict(self, audio_path: str | Path) -> dict:
        mel = self._to_logmel(Path(audio_path))
        x = torch.tensor(mel).unsqueeze(0).unsqueeze(0).float().to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        label_id = int(logits.argmax(1).item())  # 0=real, 1=fake
        label = "fake" if label_id else "real"
        return {
            "label": label,
            "is_fake": bool(label_id),
            "prob_real": float(probs[0]),
            "prob_fake": float(probs[1]),
        }


class AccentModel:
    """Stage 2: accent-proximity estimation with a wav2vec2 backbone + linear head."""

    def __init__(self, model_dir: Path = CLASSIFIER_DIR_OUT, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        with _import_context(CLASSIFIER_DIR):
            from infer import load_trained  # src/classifier/infer.py

            model, feature_extractor, labels = load_trained(Path(model_dir))
            model.to(self.device)
            # Reuse the predict function as-is for frame-level prediction (Level 2).
            from infer import predict as _predict

        self.model = model
        self.feature_extractor = feature_extractor
        self.labels = labels
        self._predict = _predict

    def predict(self, audio_path: str | Path, want_frames: bool = False) -> dict:
        probs, frame_probs = self._predict(
            self.model, self.feature_extractor, Path(audio_path), want_frames
        )
        order = np.argsort(probs)[::-1]
        ranking = [
            {"accent": self.labels[i], "percent": round(float(probs[i]) * 100, 1)}
            for i in order
        ]
        result = {
            "accents": {self.labels[i]: float(probs[i]) for i in range(len(self.labels))},
            "ranking": ranking,
            "top": self.labels[int(order[0])],
        }
        if want_frames and frame_probs is not None:
            result["frame_probs"] = frame_probs  # [T, C]
            result["frame_labels"] = list(self.labels)
        return result


class SpeechPipeline:
    """Ties the whole detector -> classifier flow together. Models are lazy-loaded on first use."""

    def __init__(self, device: str | None = None):
        self.device = device
        self._detector: DetectorModel | None = None
        self._accent: AccentModel | None = None

    @property
    def detector(self) -> DetectorModel:
        if self._detector is None:
            self._detector = DetectorModel(device=self.device)
        return self._detector

    @property
    def accent(self) -> AccentModel:
        if self._accent is None:
            self._accent = AccentModel(device=self.device)
        return self._accent

    def analyze(self, audio_path: str | Path, want_frames: bool = False) -> dict:
        """Full pipeline. The accent stage is skipped when the clip is fake."""
        det = self.detector.predict(audio_path)
        result = {"audio": str(audio_path), "detector": det}
        if det["is_fake"]:
            result["accent"] = None
            result["message"] = "Judged as AI-synthesized speech — skipping accent analysis."
        else:
            result["accent"] = self.accent.predict(audio_path, want_frames=want_frames)
        return result


def _format_human(result: dict) -> str:
    det = result["detector"]
    lines = [
        f"[detector] {det['label'].upper()}  "
        f"(real {det['prob_real'] * 100:.1f}% / fake {det['prob_fake'] * 100:.1f}%)"
    ]
    if result.get("accent") is None:
        lines.append(f"[classifier] skipped — {result.get('message', '')}")
    else:
        lines.append("[classifier] accent proximity:")
        for item in result["accent"]["ranking"]:
            lines.append(f"    {item['accent']:10s} {item['percent']:5.1f}%")
    return "\n".join(lines)


def _run_cli(audio: str, want_frames: bool, want_json: bool) -> None:
    pipe = SpeechPipeline()
    result = pipe.analyze(audio, want_frames=want_frames)

    if want_json:
        def _default(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            raise TypeError(f"not serializable: {type(o)}")

        print(json.dumps(result, ensure_ascii=False, indent=2, default=_default))
    else:
        print(_format_human(result))


# ---------------------------------------------------------------------------
# Web demo (replaces the old src/classifier/webui.py — detector + classifier combined)
# ---------------------------------------------------------------------------

# Pipeline reused by the gr.Blocks callbacks. Lazy-loaded on the first request.
_PIPE: SpeechPipeline | None = None

_PALETTE = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2"]


def _color(i: int) -> str:
    return _PALETTE[i % len(_PALETTE)]


def _heatmap_figure(frame_probs: np.ndarray, labels: list[str]):
    """Line plot of per-accent probability over time (about 20 ms per frame)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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


def _classify(audio_path: str | None):
    """Gradio callback. Returns (detector probabilities, classifier probabilities, summary md, heatmap)."""
    if not audio_path:
        return {}, {}, "🎤 Record or upload some audio.", None

    result = _PIPE.analyze(audio_path, want_frames=True)
    det = result["detector"]
    det_conf = {"real": det["prob_real"], "fake": det["prob_fake"]}

    if result["accent"] is None:
        summary = (
            f"**[detector] {det['label'].upper()}** "
            f"(real {det['prob_real'] * 100:.1f}% / fake {det['prob_fake'] * 100:.1f}%)\n\n"
            f"⛔ {result['message']}"
        )
        return det_conf, {}, summary, None

    accent = result["accent"]
    accent_conf = accent["accents"]
    lines = [
        f"**[detector] REAL** (real {det['prob_real'] * 100:.1f}% / fake {det['prob_fake'] * 100:.1f}%)",
        "",
        f"**Estimated accent: `{accent['top']}` ({accent['ranking'][0]['percent']:.1f}%)**",
        "",
        "| Accent | Proximity |",
        "|---|---|",
    ]
    for item in accent["ranking"]:
        lines.append(f"| {item['accent']} | {item['percent']:.1f}% |")
    summary = "\n".join(lines)

    fig = None
    if "frame_probs" in accent:
        fig = _heatmap_figure(accent["frame_probs"], accent["frame_labels"])

    return det_conf, accent_conf, summary, fig


def _build_demo():
    import gradio as gr

    with gr.Blocks(title="Speech Classifier") as demo:
        gr.Markdown(
            "# 🗣️ Voice Authenticity + Accent Classification (two-stage pipeline)\n"
            "Given an utterance, it first decides **1) AI-synthesized (fake) vs. real human (real)** and, "
            "only when the voice is judged human, shows **2) English accent proximity** as percentages.\n"
            "Record with the microphone or upload an audio file below, then press the **Analyze** button."
        )
        with gr.Row():
            with gr.Column(scale=1):
                audio_in = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Utterance input (record / upload)",
                )
                run_btn = gr.Button("Analyze", variant="primary")
            with gr.Column(scale=1):
                detector_out = gr.Label(label="[Stage 1] detector: real / fake", num_top_classes=2)
                accent_out = gr.Label(label="[Stage 2] classifier: accent proximity", num_top_classes=4)
        summary_out = gr.Markdown()
        heatmap_out = gr.Plot(label="Accent probability over time (Level 2, only when judged real)")

        outputs = [detector_out, accent_out, summary_out, heatmap_out]
        run_btn.click(_classify, inputs=audio_in, outputs=outputs)
        audio_in.change(_classify, inputs=audio_in, outputs=outputs)

    return demo


def _run_webui(host: str, port: int, share: bool) -> None:
    global _PIPE

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading detector + classifier onto {device} ...")
    _PIPE = SpeechPipeline(device=device)
    # Preload at startup to remove the delay on the first request.
    _ = _PIPE.detector
    _ = _PIPE.accent
    print(f"labels: {_PIPE.accent.labels}")

    demo = _build_demo()
    demo.launch(server_name=host, server_port=port, share=share)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="speech-classifier unified pipeline (CLI + web demo)")
    ap.add_argument("audio", nargs="?", default=None, help="if given, CLI mode: analyze only this audio file")
    ap.add_argument("--frames", action="store_true", help="also compute frame-level accent probabilities (Level 2, CLI mode)")
    ap.add_argument("--json", action="store_true", help="print the result as JSON (CLI mode)")
    ap.add_argument("--host", default="127.0.0.1", help="web demo host (when audio is not given)")
    ap.add_argument("--port", type=int, default=7860, help="web demo port (when audio is not given)")
    ap.add_argument("--share", action="store_true", help="create a public gradio.live link (web demo mode)")
    args = ap.parse_args()

    if args.audio is not None:
        _run_cli(args.audio, args.frames, args.json)
    else:
        _run_webui(args.host, args.port, args.share)


if __name__ == "__main__":
    main()
