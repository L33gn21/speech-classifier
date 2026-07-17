"""Dataset + collator for the accent classifier.

The Dataset yields raw 16 kHz mono waveforms (cropped to MAX_SAMPLES). The
collator runs the Wav2Vec2 feature extractor to normalize and pad each batch
to its own max length, producing `input_values` + `attention_mask`.

Clip paths resolve against config.CURATED_ROOT as
``<CURATED_ROOT>/<country>/audio/<filename>``, which points at a local dir or
a FUSE-mounted GCS bucket (``/gcs/<bucket>/curated``) on Vertex AI.
"""
# 억양 분류기용 PyTorch Dataset과 배치 콜레이터(collator) 정의.
#
# Dataset은 원본 16kHz 모노 파형을 그대로 반환하며(MAX_SAMPLES 길이로 잘림),
# 콜레이터가 Wav2Vec2 feature extractor를 실행해 각 배치를 정규화하고
# 배치 내 최대 길이에 맞춰 패딩하여 `input_values` + `attention_mask`를 만든다.
#
# 클립 파일 경로는 config.CLIPS_DIR을 기준으로 해석된다. 이 경로는 로컬
# 디렉터리이거나, Vertex AI에서는 FUSE로 마운트된 GCS 버킷 경로
# (``/gcs/<버킷>/.../clips``)일 수 있다.
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import CURATED_ROOT, MANIFEST_DIR, MAX_SAMPLES, SAMPLE_RATE


def load_audio(path: Path) -> np.ndarray:
    """Load an mp3 as float32 mono at SAMPLE_RATE. torchaudio first, librosa fallback."""
    # mp3 파일을 float32 모노 파형으로 SAMPLE_RATE(16kHz)에 맞춰 로드한다.
    # 먼저 torchaudio로 시도하고, 실패하면(코덱 미지원 등) librosa로 대체 시도한다.
    try:
        import torchaudio  # local import so data-prep doesn't need torch
        # torchaudio는 여기서만 지역 임포트한다 — 데이터 준비 단계(prepare_data.py)는
        # torch 의존성 없이도 동작해야 하기 때문.

        wav, sr = torchaudio.load(str(path))  # (channels, time)
        if wav.shape[0] > 1:
            # 스테레오 등 다채널이면 채널 평균을 내어 모노로 변환.
            wav = wav.mean(dim=0, keepdim=True)
        if sr != SAMPLE_RATE:
            # 원본 샘플링 레이트가 목표(16kHz)와 다르면 리샘플링.
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        return wav.squeeze(0).numpy().astype(np.float32)
    except Exception:
        # torchaudio 로딩 실패 시(예: 일부 mp3 인코딩 문제) librosa로 대체.
        import librosa

        wav, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
        return wav.astype(np.float32)


def augment_waveform(wav: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Light, cheap train-time augmentation: random gain + occasional noise.

    Applied only to the *train* split (never val/test). The goal is robustness
    to the channel/recording confound (GLOBE clean-24kHz vs SAA mp3, DATASET.md
    §5.1), not aggressive distortion — so gain stays mild and additive Gaussian
    noise is injected at a fairly high SNR, half the time. Waveform-level (not
    SpecAugment) so it composes with the Wav2Vec2 feature extractor downstream.
    """
    # 학습 분할에만 적용하는 가볍고 값싼 증강: 랜덤 게인 + 가끔 가우시안 노이즈.
    # 채널/녹음 confound(GLOBE 24kHz vs SAA mp3)에 대한 강건성을 노리며, 과하지 않게
    # 게인은 완만히, 노이즈는 비교적 높은 SNR로 절반 확률만 주입한다.
    wav = wav * np.float32(rng.uniform(0.8, 1.2))          # random gain
    if rng.random() < 0.5:                                  # additive noise (half the time)
        rms = float(np.sqrt(np.mean(wav ** 2) + 1e-9))
        snr_db = rng.uniform(15.0, 30.0)
        noise_rms = rms / (10.0 ** (snr_db / 20.0))
        wav = wav + rng.normal(0.0, noise_rms, size=wav.shape).astype(np.float32)
    return wav.astype(np.float32)


class AccentDataset(Dataset):
    def __init__(
        self,
        manifest: "str | Path | pd.DataFrame",
        curated_root: Path = CURATED_ROOT,
        augment: bool = False,
    ):
        # manifest: filename,label,country[,speaker,source] 컬럼을 가진 CSV 경로이거나
        # 이미 로드된 DataFrame (prepare_data.build_splits 가 만든 train/val/test).
        # 오디오는 <curated_root>/<country>/audio/<filename> 에서 로드한다.
        # augment: True 면 파형 증강을 적용한다(학습 분할에만 켤 것).
        if isinstance(manifest, pd.DataFrame):
            self.df = manifest.reset_index(drop=True)
        else:
            self.df = pd.read_csv(manifest)
        self.curated_root = Path(curated_root)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        path = self.curated_root / row["country"] / "audio" / row["filename"]
        wav = load_audio(path)
        if len(wav) > MAX_SAMPLES:
            wav = wav[:MAX_SAMPLES]  # crop long clips
            # 너무 긴 클립은 앞부분 MAX_SAMPLES(기본 8초)만 잘라서 사용한다.
        if self.augment:
            # default_rng() (seed 없음)는 OS 엔트로피로 초기화되어 DataLoader
            # 워커/호출마다 독립적이다 — 증강은 재현성이 필요없다.
            wav = augment_waveform(wav, np.random.default_rng())
        return {"waveform": wav, "label": int(row["label"])}


@dataclass
class DataCollator:
    """Normalize + pad a batch via the Wav2Vec2 feature extractor."""
    # DataLoader가 배치를 만들 때 호출되는 콜레이터.
    # Wav2Vec2FeatureExtractor를 이용해 배치 내 파형들을 정규화하고,
    # 배치 안에서 가장 긴 길이에 맞춰 패딩한다.

    feature_extractor: object  # transformers Wav2Vec2FeatureExtractor

    def __call__(self, batch: list[dict]) -> dict:
        waveforms = [b["waveform"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        out = self.feature_extractor(
            waveforms,
            sampling_rate=SAMPLE_RATE,
            padding=True,               # 배치 내 최대 길이까지 패딩
            return_attention_mask=True, # 패딩 위치를 모델에 알려주기 위한 마스크 생성
            return_tensors="pt",
        )
        out["labels"] = labels
        return out


def get_datasets(manifest_dir: Path = MANIFEST_DIR):
    # 이미 기록된 train/val/test.csv 매니페스트로부터 세 데이터셋을 만들어 반환.
    train = AccentDataset(Path(manifest_dir) / "train.csv")
    val = AccentDataset(Path(manifest_dir) / "val.csv")
    test = AccentDataset(Path(manifest_dir) / "test.csv")
    return train, val, test
