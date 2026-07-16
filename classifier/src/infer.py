"""Run a trained accent classifier on an audio file.

Outputs per-accent proximity percentages (Level 1). With --frames it also
returns frame-level probabilities (Level 2 time-axis heatmap material).

Example:
    python infer.py path/to/clip.mp3
    python infer.py clip.mp3 --frames --plot heatmap.png
"""
# 학습된 억양 분류기를 단일 오디오 파일에 대해 실행(추론)하는 스크립트.
#
# 기본적으로 각 억양 클래스에 대한 근접도(확률)를 퍼센트로 출력한다(Level 1).
# --frames 옵션을 주면 프레임(시간) 단위 확률까지 함께 반환하며, 이는
# Level 2에서 사용할 "시간축 억양 히트맵"의 재료가 된다.
#
# 실행 예시:
#     python infer.py path/to/clip.mp3
#     python infer.py clip.mp3 --frames --plot heatmap.png
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
    # 저장된 모델 디렉터리로부터 모델 가중치, feature extractor, 레이블 목록을
    # 불러온다. label_config.json이 있으면 그 안의 레이블 순서를 우선 사용한다
    # (혹시 학습 당시 config.LABELS와 달라졌을 경우를 대비).
    model = AccentClassifier(MODEL_NAME)
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
    # 단일 오디오 파일을 로드하여 전처리한 뒤 모델에 통과시키고,
    # 발화 전체 확률(및 필요 시 프레임별 확률)을 반환한다.
    wav = load_audio(audio_path)
    inputs = feature_extractor(
        [wav], sampling_rate=SAMPLE_RATE, return_attention_mask=True, return_tensors="pt"
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model(**inputs, output_frame_logits=want_frames)
    # softmax로 로짓을 확률 분포로 변환. 배치 크기가 1이므로 [0]으로 꺼냄.
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
    # 프레임 단위 확률도 함께 계산할지 여부.
    ap.add_argument("--plot", default=None, help="save a frame-level heatmap PNG (implies --frames)")
    # 히트맵 PNG를 저장할 경로. 지정하면 자동으로 --frames도 활성화된 것으로 간주.
    args = ap.parse_args()

    model, fe, labels = load_trained(Path(args.model_dir))
    want_frames = args.frames or args.plot is not None
    probs, frame_probs = predict(model, fe, Path(args.audio), want_frames)

    # 확률이 높은 순서대로 정렬해서 출력 (가장 가능성 높은 억양이 먼저 나옴).
    order = np.argsort(probs)[::-1]
    print(f"\nAccent proximity for {args.audio}:")
    for i in order:
        print(f"  {labels[i]:10s} {probs[i] * 100:5.1f}%")

    if args.plot is not None:
        import matplotlib

        matplotlib.use("Agg")  # GUI 없는 환경(서버 등)에서도 동작하도록 백엔드 고정
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 3))
        # ~20ms per wav2vec2 frame; x in seconds
        # wav2vec2의 프레임 하나는 대략 20ms에 해당하므로, 이를 이용해
        # 프레임 인덱스를 실제 시간(초)으로 환산한다.
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
