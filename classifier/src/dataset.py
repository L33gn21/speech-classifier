"""Dataset + collator for the accent classifier.

The Dataset yields raw 16 kHz mono waveforms (cropped to MAX_SAMPLES). The
collator runs the Wav2Vec2 feature extractor to normalize and pad each batch
to its own max length, producing `input_values` + `attention_mask`.

Clip paths resolve against config.CURATED_ROOT as
``<CURATED_ROOT>/<country>/audio/<filename>``, which points at a local dir or
a FUSE-mounted GCS bucket (``/gcs/<bucket>/curated``) on Vertex AI.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import (
    CURATED_ROOT,
    MANIFEST_DIR,
    MAX_SAMPLES,
    SAMPLE_RATE,
    gcs_to_fuse,
)


def load_audio(path: Path) -> np.ndarray:
    """Load an mp3 as float32 mono at SAMPLE_RATE. torchaudio first, librosa fallback."""
    try:
        import torchaudio  # local import so data-prep doesn't need torch

        wav, sr = torchaudio.load(str(path))  # (channels, time)
        if wav.shape[0] > 1:
            # Multi-channel (e.g. stereo) input: average the channels down to mono.
            wav = wav.mean(dim=0, keepdim=True)
        if sr != SAMPLE_RATE:
            # Resample when the source sampling rate differs from the target (16 kHz).
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        return wav.squeeze(0).numpy().astype(np.float32)
    except Exception:
        # Fall back to librosa when torchaudio fails to load (e.g. some mp3 encoding issues).
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

    This is the *legacy* (v3) augmentation, kept unchanged so the v3 recipe is
    exactly reproducible. For domain-shift robustness use ``domain_augment``
    (aug_strength > 0), which simulates realistic recording conditions.
    """
    wav = wav * np.float32(rng.uniform(0.8, 1.2))          # random gain
    if rng.random() < 0.5:                                  # additive noise (half the time)
        rms = float(np.sqrt(np.mean(wav ** 2) + 1e-9))
        snr_db = rng.uniform(15.0, 30.0)
        noise_rms = rms / (10.0 ** (snr_db / 20.0))
        wav = wav + rng.normal(0.0, noise_rms, size=wav.shape).astype(np.float32)
    return wav.astype(np.float32)


# --- domain-randomization augmentation (v4: GLOBE -> VoxForge domain gap) ------
# Why: the biggest remaining v3 issue is the CA→US collapse on an unseen corpus (VoxForge).
# The diagnosis (reports/2026-07-18-channel-leakage-probe.md) attributed it to "genuine
# accent similarity + domain shift, not channel leakage". The training data (GLOBE) is
# clean, TTS-grade 24 kHz audio, while the evaluation target (VoxForge) is amateur home
# recording (limited bandwidth, reverb, real-world noise, varying speaking rate).
# "Domain randomization" at training time mimics this gap and reduces overfitting to the
# source domain (domain adaptation, countermeasure C).
# This is a completely different axis from the legacy augmentation (gain + Gaussian) and
# from strong SpecAugment (feature masking): waveform-level distortion that reproduces
# realistic recording conditions.
# Everything uses numpy only (cheap on CPU, composed in front of the feature extractor).


def _windowed_sinc_lowpass(cutoff_hz: float, sr: int, num_taps: int = 63) -> np.ndarray:
    """Design a simple windowed-sinc FIR low-pass kernel (Hamming window)."""
    # Mimics limited bandwidth (amateur microphones / telephone networks).
    # Designed with numpy only, without scipy.
    fc = np.clip(cutoff_hz / sr, 1e-3, 0.5 - 1e-3)  # normalized cutoff (cycles/sample)
    n = np.arange(num_taps) - (num_taps - 1) / 2.0
    h = 2 * fc * np.sinc(2 * fc * n)                # ideal sinc
    h *= np.hamming(num_taps)                       # window to tame ringing
    h /= h.sum()                                    # unit DC gain
    return h.astype(np.float32)


def _synthetic_reverb_ir(rng: np.random.Generator, sr: int, strength: float) -> np.ndarray:
    """Short exponentially-decaying synthetic room impulse response."""
    # Mimics room reverberation. Provides the domain signal as "direct sound + decaying
    # reverb tail" without needing a real RIR library.
    rt60 = rng.uniform(0.10, 0.10 + 0.35 * strength)         # decay time (seconds)
    length = max(8, int(sr * rt60))
    t = np.arange(length)
    decay = np.exp(-6.9 * t / length)                        # -60 dB at the tail
    # The reverb tail is quieter than the direct sound (≈ -10 dB). No energy normalization
    # is applied, so ir[0]=1 is kept and the convolution yields "original + decaying reverb"
    # (the original is preserved; the wet/dry mix controls the strength).
    ir = (rng.standard_normal(length) * decay * 0.3).astype(np.float32)
    ir[0] = 1.0                                              # direct path (dominant)
    return ir


def domain_augment(wav: np.ndarray, rng: np.random.Generator,
                   strength: float = 1.0) -> np.ndarray:
    """Realistic recording-condition randomization to close the GLOBE->VoxForge gap.

    Each perturbation fires with its own probability (scaled by ``strength`` in
    [0, 1+]) and randomizes toward the amateur-home-recording domain the model
    generalizes poorly to. All waveform-level and numpy-only so it composes with
    the Wav2Vec2 feature extractor and stays cheap on the dataloader CPU workers.
    Order mirrors a real capture chain: speed -> band-limit -> reverb -> gain ->
    noise. Returns a finite float32 waveform (peak-limited to avoid clipping).

    strength=0 is a no-op (caller should use the legacy path instead); 1.0 is the
    default full-strength preset validated in the aug-strength sweep.
    """
    if strength <= 0:
        return wav.astype(np.float32)
    s = float(strength)
    x = wav.astype(np.float32)

    # 1) speed perturbation (± speed) — speaking-rate/pitch variation. np.interp resampling (cheap).
    if rng.random() < 0.5 * min(s, 1.0):
        rate = float(rng.uniform(1.0 - 0.10 * s, 1.0 + 0.10 * s))
        if abs(rate - 1.0) > 1e-3 and len(x) > 4:
            new_len = max(4, int(round(len(x) / rate)))
            src = np.linspace(0.0, len(x) - 1, num=new_len, dtype=np.float32)
            x = np.interp(src, np.arange(len(x), dtype=np.float32), x).astype(np.float32)

    # 2) band-limiting low-pass — limited-bandwidth microphones / telephone networks.
    #    FIR with a random cutoff.
    if rng.random() < 0.5 * min(s, 1.0):
        cutoff = float(rng.uniform(3200.0, 7200.0))
        h = _windowed_sinc_lowpass(cutoff, SAMPLE_RATE)
        x = np.convolve(x, h, mode="same").astype(np.float32)

    # 3) reverb — room reverberation. Convolve with a short synthetic RIR, then mix wet/dry.
    if rng.random() < 0.35 * min(s, 1.0):
        ir = _synthetic_reverb_ir(rng, SAMPLE_RATE, s)
        wet = np.convolve(x, ir, mode="full")[: len(x)].astype(np.float32)
        mix = float(rng.uniform(0.15, 0.15 + 0.45 * s))
        x = ((1.0 - mix) * x + mix * wet).astype(np.float32)

    # 4) random gain — variation in microphone distance / input gain (wider than legacy).
    x = x * np.float32(rng.uniform(1.0 - 0.4 * s, 1.0 + 0.4 * s))

    # 5) additive noise — real-world background sound. Half the time, over a wide and low
    #    SNR range. Half of it is low-passed into 'colored' noise (closer to home-recording
    #    background sound than white noise).
    if rng.random() < 0.6 * min(s, 1.0):
        rms = float(np.sqrt(np.mean(x ** 2) + 1e-9))
        snr_db = float(rng.uniform(30.0 - 22.0 * s, 30.0 - 5.0 * s))
        noise_rms = rms / (10.0 ** (snr_db / 20.0))
        noise = rng.normal(0.0, noise_rms, size=x.shape).astype(np.float32)
        if rng.random() < 0.5:
            noise = np.convolve(
                noise, _windowed_sinc_lowpass(rng.uniform(2000.0, 6000.0), SAMPLE_RATE),
                mode="same").astype(np.float32)
        x = x + noise

    # peak-limit so downstream normalization sees a sane range (avoid hard clip).
    peak = float(np.max(np.abs(x)) + 1e-9)
    if peak > 1.0:
        x = x / peak
    return np.nan_to_num(x, copy=False).astype(np.float32)


class AccentDataset(Dataset):
    def __init__(
        self,
        manifest: "str | Path | pd.DataFrame",
        curated_root: Path = CURATED_ROOT,
        augment: bool = False,
        aug_strength: float = 0.0,
    ):
        # manifest: path to a CSV with the columns filename,label,country[,speaker,source], or
        # an already loaded DataFrame (train/val/test built by prepare_data.build_splits).
        # Audio is loaded from <curated_root>/<country>/audio/<filename>.
        # augment: if True, waveform augmentation is applied (enable for the train split only).
        # aug_strength: 0 = legacy (v3) light augmentation (gain + Gaussian); >0 = domain
        #   randomization (domain_augment) — the larger the value, the more strongly the
        #   GLOBE→VoxForge domain gap is mimicked (countermeasure C). Ignored when augment=False.
        if isinstance(manifest, pd.DataFrame):
            self.df = manifest.reset_index(drop=True)
        else:
            self.df = pd.read_csv(manifest)
        self.curated_root = Path(curated_root)
        self.augment = augment
        self.aug_strength = float(aug_strength)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        path = self.curated_root / row["country"] / "audio" / row["filename"]
        wav = load_audio(path)
        # default_rng() (no seed) is initialized from OS entropy and is independent per
        # DataLoader worker/call — augmentation does not need to be reproducible.
        rng = np.random.default_rng() if self.augment else None
        if len(wav) > MAX_SAMPLES:
            if self.augment:
                # In training, crop a MAX_SAMPLES window at a random position (free augmentation, and
                # wider coverage so long clips/SAA paragraphs are not biased toward their beginning).
                start = int(rng.integers(0, len(wav) - MAX_SAMPLES + 1))
                wav = wav[start:start + MAX_SAMPLES]
            else:
                # In evaluation, deterministically use only the first MAX_SAMPLES (8 s by default).
                wav = wav[:MAX_SAMPLES]
        if self.augment:
            if self.aug_strength > 0:
                # Domain randomization (countermeasure C). Speed perturbation can lengthen the clip, so
                # crop again to the training-window limit (MAX_SAMPLES) to avoid wasted batch padding.
                wav = domain_augment(wav, rng, self.aug_strength)
                if len(wav) > MAX_SAMPLES:
                    wav = wav[:MAX_SAMPLES]
            else:
                wav = augment_waveform(wav, rng)          # legacy (v3) light augmentation
        return {"waveform": wav, "label": int(row["label"])}


def _crop_and_augment(wav: np.ndarray, augment: bool, aug_strength: float) -> np.ndarray:
    """Shared crop (+ optional augmentation) — identical policy to AccentDataset.

    Kept as a module helper so MultiTaskDataset applies the *exact same* window
    crop and augmentation (legacy or domain-randomization) that the validated
    single-task path uses. real and fake clips go through this identically, which
    is the whole point of the multi-task channel-confound control (both sides get
    the same domain randomization so channel can't be a shortcut).
    """
    rng = np.random.default_rng() if augment else None
    if len(wav) > MAX_SAMPLES:
        if augment:
            start = int(rng.integers(0, len(wav) - MAX_SAMPLES + 1))
            wav = wav[start:start + MAX_SAMPLES]
        else:
            wav = wav[:MAX_SAMPLES]
    if augment:
        if aug_strength > 0:
            wav = domain_augment(wav, rng, aug_strength)
            if len(wav) > MAX_SAMPLES:
                wav = wav[:MAX_SAMPLES]
        else:
            wav = augment_waveform(wav, rng)
    return wav


class MultiTaskDataset(Dataset):
    """Dataset for joint country + real/fake training over a unified manifest.

    The unified manifest (built by prepare_data_multitask.build_multitask_splits,
    DATASET.md §11) has columns: ``filename, audio_uri, country, country_label,
    fake_label, speaker, source, system_id, orig_split``. ``audio_uri`` is
    already a full ``gs://`` (or local) path — real country-sourced rows point
    at ``curated/<CC>/audio/``, ASVspoof-derived/oversample-dup rows point at
    ``curated_spoof/real_fake_5k/audio_asv|audio_dup/`` — so each row resolves
    its own audio independently; no shared root is needed.

    ``country_label`` is the 0..5 country id for country-sourced clips, or
    ``COUNTRY_IGNORE_INDEX`` (-100) for ASVspoof-sourced clips (no country
    label -> ignored by the country loss). ``fake_label`` is 0=real / 1=fake
    for every clip.
    """
    def __init__(
        self,
        manifest: "str | Path | pd.DataFrame",
        augment: bool = False,
        aug_strength: float = 0.0,
    ):
        if isinstance(manifest, pd.DataFrame):
            self.df = manifest.reset_index(drop=True)
        else:
            self.df = pd.read_csv(manifest)
        self.augment = augment
        self.aug_strength = float(aug_strength)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        path = Path(gcs_to_fuse(str(row["audio_uri"])))
        wav = load_audio(path)
        wav = _crop_and_augment(wav, self.augment, self.aug_strength)
        return {
            "waveform": wav,
            "country_label": int(row["country_label"]),
            "fake_label": int(row["fake_label"]),
        }


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
            padding=True,               # pad to the longest clip in the batch
            return_attention_mask=True, # create the mask that tells the model where the padding is
            return_tensors="pt",
        )
        out["labels"] = labels
        return out


@dataclass
class MultiTaskCollator:
    """Normalize + pad a batch and emit *two* label tensors (country + fake).

    Mirrors DataCollator but returns ``country_labels`` and ``fake_labels`` under
    those exact keys so the HF Trainer (with
    ``TrainingArguments.label_names=["country_labels","fake_labels"]``) forwards
    both to the model and back out of ``predict()`` as a label tuple.
    """
    feature_extractor: object

    def __call__(self, batch: list[dict]) -> dict:
        waveforms = [b["waveform"] for b in batch]
        out = self.feature_extractor(
            waveforms,
            sampling_rate=SAMPLE_RATE,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        out["country_labels"] = torch.tensor(
            [b["country_label"] for b in batch], dtype=torch.long)
        out["fake_labels"] = torch.tensor(
            [b["fake_label"] for b in batch], dtype=torch.long)
        return out


def get_datasets(manifest_dir: Path = MANIFEST_DIR):
    # Build and return the three datasets from the already written train/val/test.csv manifests.
    train = AccentDataset(Path(manifest_dir) / "train.csv")
    val = AccentDataset(Path(manifest_dir) / "val.csv")
    test = AccentDataset(Path(manifest_dir) / "test.csv")
    return train, val, test
