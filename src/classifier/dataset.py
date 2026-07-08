"""Dataset + collator for the accent classifier.

The Dataset yields raw 16 kHz mono waveforms (cropped to MAX_SAMPLES). The
collator runs the Wav2Vec2 feature extractor to normalize and pad each batch
to its own max length, producing `input_values` + `attention_mask`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import CLIPS_DIR, MANIFEST_DIR, MAX_SAMPLES, SAMPLE_RATE


def load_audio(path: Path) -> np.ndarray:
    """Load an mp3 as float32 mono at SAMPLE_RATE. torchaudio first, librosa fallback."""
    try:
        import torchaudio  # local import so data-prep doesn't need torch

        wav, sr = torchaudio.load(str(path))  # (channels, time)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        return wav.squeeze(0).numpy().astype(np.float32)
    except Exception:
        import librosa

        wav, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
        return wav.astype(np.float32)


class AccentDataset(Dataset):
    def __init__(self, manifest_csv: str | Path, clips_dir: Path = CLIPS_DIR):
        self.df = pd.read_csv(manifest_csv)
        self.clips_dir = Path(clips_dir)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        wav = load_audio(self.clips_dir / row["filename"])
        if len(wav) > MAX_SAMPLES:
            wav = wav[:MAX_SAMPLES]  # crop long clips
        return {"waveform": wav, "label": int(row["label"])}


@dataclass
class DataCollator:
    """Normalize + pad a batch via the Wav2Vec2 feature extractor."""

    feature_extractor: object  # transformers Wav2Vec2FeatureExtractor

    def __call__(self, batch: list[dict]) -> dict:
        waveforms = [b["waveform"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        out = self.feature_extractor(
            waveforms,
            sampling_rate=SAMPLE_RATE,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        out["labels"] = labels
        return out


def get_datasets(manifest_dir: Path = MANIFEST_DIR):
    train = AccentDataset(manifest_dir / "train.csv")
    test = AccentDataset(manifest_dir / "test.csv")
    return train, test
