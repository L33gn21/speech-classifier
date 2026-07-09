"""음성 파이프라인 통합 진입점 (detector -> classifier).

발화 하나를 받아 2단계 파이프라인을 돌린다.
  1) detector : AI 합성(fake) vs 실제 사람(real) 판별
  2) classifier: real 로 판정된 경우에만 영어 억양권 근접도(%) 추정

detector 와 classifier 는 각자 `from model import ...` / `from config import ...`
같은 플랫 import 를 쓰고, 두 디렉터리 모두 model.py/config.py/dataset.py 를 갖는다.
그래서 두 패키지를 동시에 sys.path 에 올리면 이름이 충돌한다. 아래 `_import_context`
컨텍스트 매니저로 한 번에 한 패키지만 import 경로에 노출시키고, 충돌하는 모듈
캐시를 정리해 각 서브시스템을 독립적으로 로드한다.

Python API:
    from app import SpeechPipeline
    pipe = SpeechPipeline()
    print(pipe.analyze("clip.wav"))

CLI (단일 파일 분석):
    .venv/bin/python src/app.py path/to/clip.wav [--frames] [--json]

웹 데모 (마이크/업로드 -> detector 판정 + classifier 억양 % + 프레임 히트맵):
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

# 두 서브패키지가 공유하는 플랫 모듈 이름들 — import 전에 캐시에서 비운다.
_CONFLICTING_MODULES = ("config", "model", "dataset", "infer", "inference")


@contextlib.contextmanager
def _import_context(pkg_dir: Path):
    """`pkg_dir` 만 sys.path 앞에 올리고 충돌 모듈 캐시를 정리한 상태로 import 하게 한다."""
    saved_path = list(sys.path)
    saved_modules = {name: sys.modules.pop(name) for name in _CONFLICTING_MODULES if name in sys.modules}
    sys.path.insert(0, str(pkg_dir))
    try:
        yield
    finally:
        sys.path[:] = saved_path
        # 이번 컨텍스트에서 새로 로드된 플랫 모듈을 제거하고, 원래 있던 것은 복원.
        for name in _CONFLICTING_MODULES:
            sys.modules.pop(name, None)
        sys.modules.update(saved_modules)


class DetectorModel:
    """1단계: log-mel + resnet18 로 real/fake 판별."""

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
        """inference.py 와 동일한 전처리: 16kHz -> 128-mel log, 128프레임 crop/pad, 표준화."""
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
    """2단계: wav2vec2 backbone + linear head 로 억양 근접도 추정."""

    def __init__(self, model_dir: Path = CLASSIFIER_DIR_OUT, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        with _import_context(CLASSIFIER_DIR):
            from infer import load_trained  # src/classifier/infer.py

            model, feature_extractor, labels = load_trained(Path(model_dir))
            model.to(self.device)
            # 프레임 단위 예측(레벨 2)에 predict 함수를 그대로 재사용한다.
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
    """detector -> classifier 전체 흐름을 묶는다. 모델은 첫 사용 시 lazy 로드."""

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
        """전체 파이프라인. fake 면 억양 단계를 건너뛴다."""
        det = self.detector.predict(audio_path)
        result = {"audio": str(audio_path), "detector": det}
        if det["is_fake"]:
            result["accent"] = None
            result["message"] = "AI 합성 음성으로 판정 — 억양 분석을 건너뜁니다."
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
        lines.append("[classifier] 억양 근접도:")
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
# 웹 데모 (구 src/classifier/webui.py 를 대체 — detector + classifier 통합)
# ---------------------------------------------------------------------------

# gr.Blocks 콜백에서 재사용할 파이프라인. 첫 요청 시 lazy 로드.
_PIPE: SpeechPipeline | None = None

_PALETTE = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2"]


def _color(i: int) -> str:
    return _PALETTE[i % len(_PALETTE)]


def _heatmap_figure(frame_probs: np.ndarray, labels: list[str]):
    """시간축(약 20ms/프레임) 억양별 확률 라인 플롯."""
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
    """Gradio 콜백. (detector 확률, classifier 확률, 요약 md, 히트맵) 반환."""
    if not audio_path:
        return {}, {}, "🎤 오디오를 녹음하거나 업로드해줘.", None

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
        f"**추정 억양: `{accent['top']}` ({accent['ranking'][0]['percent']:.1f}%)**",
        "",
        "| 억양 | 근접도 |",
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
            "# 🗣️ 음성 판별 + 억양 분류 (2단계 파이프라인)\n"
            "발화를 입력하면 **1) AI 합성(fake) vs 실제 사람(real)** 을 먼저 판별하고, "
            "사람 음성으로 판정된 경우에만 **2) 영어 억양 근접도** 를 퍼센트로 보여줘.\n"
            "아래에서 마이크로 녹음하거나 오디오 파일을 올리고 **분석** 버튼을 눌러."
        )
        with gr.Row():
            with gr.Column(scale=1):
                audio_in = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="발화 입력 (녹음 / 업로드)",
                )
                run_btn = gr.Button("분석", variant="primary")
            with gr.Column(scale=1):
                detector_out = gr.Label(label="[1단계] detector: real / fake", num_top_classes=2)
                accent_out = gr.Label(label="[2단계] classifier: 억양 근접도", num_top_classes=4)
        summary_out = gr.Markdown()
        heatmap_out = gr.Plot(label="시간축 억양 확률 (Level 2, real 판정 시에만)")

        outputs = [detector_out, accent_out, summary_out, heatmap_out]
        run_btn.click(_classify, inputs=audio_in, outputs=outputs)
        audio_in.change(_classify, inputs=audio_in, outputs=outputs)

    return demo


def _run_webui(host: str, port: int, share: bool) -> None:
    global _PIPE

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading detector + classifier onto {device} ...")
    _PIPE = SpeechPipeline(device=device)
    # 시작 시 미리 로드해서 첫 요청 지연을 없앤다.
    _ = _PIPE.detector
    _ = _PIPE.accent
    print(f"labels: {_PIPE.accent.labels}")

    demo = _build_demo()
    demo.launch(server_name=host, server_port=port, share=share)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="speech-classifier 통합 파이프라인 (CLI + 웹 데모)")
    ap.add_argument("audio", nargs="?", default=None, help="지정 시 CLI 모드: 해당 오디오 파일만 분석")
    ap.add_argument("--frames", action="store_true", help="프레임 단위 억양 확률도 계산(레벨 2, CLI 모드)")
    ap.add_argument("--json", action="store_true", help="결과를 JSON 으로 출력(CLI 모드)")
    ap.add_argument("--host", default="127.0.0.1", help="웹 데모 host (audio 미지정 시)")
    ap.add_argument("--port", type=int, default=7860, help="웹 데모 port (audio 미지정 시)")
    ap.add_argument("--share", action="store_true", help="공개 gradio.live 링크 생성 (웹 데모 모드)")
    args = ap.parse_args()

    if args.audio is not None:
        _run_cli(args.audio, args.frames, args.json)
    else:
        _run_webui(args.host, args.port, args.share)


if __name__ == "__main__":
    main()
