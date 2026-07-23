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

from config import (
    CURATED_ROOT,
    MANIFEST_DIR,
    MAX_SAMPLES,
    SAMPLE_RATE,
    gcs_to_fuse,
)


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

    This is the *legacy* (v3) augmentation, kept unchanged so the v3 recipe is
    exactly reproducible. For domain-shift robustness use ``domain_augment``
    (aug_strength > 0), which simulates realistic recording conditions.
    """
    # 학습 분할에만 적용하는 가볍고 값싼 증강: 랜덤 게인 + 가끔 가우시안 노이즈.
    # 채널/녹음 confound(GLOBE 24kHz vs SAA mp3)에 대한 강건성을 노리며, 과하지 않게
    # 게인은 완만히, 노이즈는 비교적 높은 SNR로 절반 확률만 주입한다.
    # (이건 v3 레거시 증강 — 정확한 재현을 위해 그대로 둔다. 도메인 시프트 강건성은
    #  아래 domain_augment(aug_strength>0)를 쓴다.)
    wav = wav * np.float32(rng.uniform(0.8, 1.2))          # random gain
    if rng.random() < 0.5:                                  # additive noise (half the time)
        rms = float(np.sqrt(np.mean(wav ** 2) + 1e-9))
        snr_db = rng.uniform(15.0, 30.0)
        noise_rms = rms / (10.0 ** (snr_db / 20.0))
        wav = wav + rng.normal(0.0, noise_rms, size=wav.shape).astype(np.float32)
    return wav.astype(np.float32)


# --- domain-randomization augmentation (v4: GLOBE -> VoxForge domain gap) ------
# 왜: v3의 최대 잔여 과제 = 미지 코퍼스(VoxForge)에서 CA→US 붕괴. 진단(reports/
# 2026-07-18-channel-leakage-probe.md)이 원인을 "채널 누수가 아닌 진짜 억양 유사성 +
# 도메인 시프트"로 판정했다. 학습 데이터(GLOBE)는 깨끗한 24kHz TTS급이고, 평가
# 타깃(VoxForge)은 아마추어 홈레코딩(대역폭 제한·잔향·실환경 노이즈·발화속도 변동)이다.
# 이 갭을 학습 시 "도메인 랜덤화"로 흉내내 소스 도메인 과적합을 줄인다(도메인 적응, 대책 C).
# 레거시(gain+가우시안)와 강한 SpecAugment(피처 마스킹)와는 완전히 다른 축 — 파형 레벨의
# 현실적 녹음조건 왜곡이다. 전부 numpy만 사용(CPU 값쌈, feature extractor 앞단에 합성).


def _windowed_sinc_lowpass(cutoff_hz: float, sr: int, num_taps: int = 63) -> np.ndarray:
    """Design a simple windowed-sinc FIR low-pass kernel (Hamming window)."""
    # 대역폭 제한(아마추어 마이크/전화망)을 흉내내기 위한 간단한 윈도우드-싱크 FIR
    # 저역통과 커널. scipy 없이 numpy만으로 설계한다.
    fc = np.clip(cutoff_hz / sr, 1e-3, 0.5 - 1e-3)  # normalized cutoff (cycles/sample)
    n = np.arange(num_taps) - (num_taps - 1) / 2.0
    h = 2 * fc * np.sinc(2 * fc * n)                # ideal sinc
    h *= np.hamming(num_taps)                       # window to tame ringing
    h /= h.sum()                                    # unit DC gain
    return h.astype(np.float32)


def _synthetic_reverb_ir(rng: np.random.Generator, sr: int, strength: float) -> np.ndarray:
    """Short exponentially-decaying synthetic room impulse response."""
    # 짧은 지수감쇠 합성 룸 임펄스 응답(RIR) — 방 잔향을 흉내낸다. 실제 RIR 라이브러리
    # 없이도 "직접음 + 감쇠 반향 꼬리"로 도메인 신호를 준다.
    rt60 = rng.uniform(0.10, 0.10 + 0.35 * strength)         # 감쇠 시간(초)
    length = max(8, int(sr * rt60))
    t = np.arange(length)
    decay = np.exp(-6.9 * t / length)                        # -60 dB at the tail
    # 반향 꼬리는 직접음보다 조용하게(≈-10 dB). 에너지 정규화를 하지 않으므로 ir[0]=1
    # 이 유지되어 conv 결과가 "원음 + 감쇠 반향"이 된다(원음 보존, wet/dry 혼합이 세기 제어).
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
    # 현실적 녹음조건 랜덤화로 GLOBE→VoxForge 도메인 갭을 좁힌다. 각 왜곡은 자체
    # 확률(strength로 스케일)로 발동하며, 실제 캡처 체인 순서(속도→대역제한→잔향→
    # 게인→노이즈)를 따른다. 전부 파형 레벨·numpy 전용이라 값싸고 feature extractor
    # 앞단에 자연히 합성된다. strength=0이면 아무것도 안 함(호출측이 레거시 경로 사용).
    if strength <= 0:
        return wav.astype(np.float32)
    s = float(strength)
    x = wav.astype(np.float32)

    # 1) speed perturbation (±속도) — 발화 속도/피치 변동. np.interp 리샘플(값쌈).
    if rng.random() < 0.5 * min(s, 1.0):
        rate = float(rng.uniform(1.0 - 0.10 * s, 1.0 + 0.10 * s))
        if abs(rate - 1.0) > 1e-3 and len(x) > 4:
            new_len = max(4, int(round(len(x) / rate)))
            src = np.linspace(0.0, len(x) - 1, num=new_len, dtype=np.float32)
            x = np.interp(src, np.arange(len(x), dtype=np.float32), x).astype(np.float32)

    # 2) band-limiting low-pass — 제한된 대역폭 마이크/전화망. 랜덤 컷오프 FIR.
    if rng.random() < 0.5 * min(s, 1.0):
        cutoff = float(rng.uniform(3200.0, 7200.0))
        h = _windowed_sinc_lowpass(cutoff, SAMPLE_RATE)
        x = np.convolve(x, h, mode="same").astype(np.float32)

    # 3) reverb — 방 잔향. 짧은 합성 RIR과 컨볼브 후 wet/dry 혼합.
    if rng.random() < 0.35 * min(s, 1.0):
        ir = _synthetic_reverb_ir(rng, SAMPLE_RATE, s)
        wet = np.convolve(x, ir, mode="full")[: len(x)].astype(np.float32)
        mix = float(rng.uniform(0.15, 0.15 + 0.45 * s))
        x = ((1.0 - mix) * x + mix * wet).astype(np.float32)

    # 4) random gain — 마이크 거리/입력 게인 변동(레거시보다 넓게).
    x = x * np.float32(rng.uniform(1.0 - 0.4 * s, 1.0 + 0.4 * s))

    # 5) additive noise — 실환경 배경음. 절반 확률로, 넓고 낮은 SNR. 절반은 저역통과
    #    시켜 '컬러' 노이즈(백색보다 홈레코딩 배경음에 가깝게).
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
    # 게인/노이즈 후 피크가 튀면 다운스트림 정규화가 왜곡되므로 부드럽게 리미팅.
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
        # manifest: filename,label,country[,speaker,source] 컬럼을 가진 CSV 경로이거나
        # 이미 로드된 DataFrame (prepare_data.build_splits 가 만든 train/val/test).
        # 오디오는 <curated_root>/<country>/audio/<filename> 에서 로드한다.
        # augment: True 면 파형 증강을 적용한다(학습 분할에만 켤 것).
        # aug_strength: 0 이면 레거시(v3) 경량 증강(gain+가우시안), >0 이면 도메인
        #   랜덤화(domain_augment) — 값이 클수록 GLOBE→VoxForge 도메인 갭을 강하게
        #   흉내낸다(대책 C). augment=False 면 무시된다.
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
        # default_rng() (seed 없음)는 OS 엔트로피로 초기화되어 DataLoader
        # 워커/호출마다 독립적이다 — 증강은 재현성이 필요없다.
        rng = np.random.default_rng() if self.augment else None
        if len(wav) > MAX_SAMPLES:
            if self.augment:
                # 학습 시엔 무작위 위치의 MAX_SAMPLES 윈도를 잘라 쓴다(공짜 증강 +
                # 긴 클립/SAA 문단의 앞부분에만 치우치지 않게 커버리지 확대).
                start = int(rng.integers(0, len(wav) - MAX_SAMPLES + 1))
                wav = wav[start:start + MAX_SAMPLES]
            else:
                # 평가 시엔 결정적으로 앞부분 MAX_SAMPLES(기본 8초)만 사용한다.
                wav = wav[:MAX_SAMPLES]
        if self.augment:
            if self.aug_strength > 0:
                # 도메인 랜덤화(대책 C). 속도 왜곡으로 길이가 늘 수 있으니 학습 윈도우
                # 상한(MAX_SAMPLES)으로 다시 잘라 배치 패딩 낭비를 막는다.
                wav = domain_augment(wav, rng, self.aug_strength)
                if len(wav) > MAX_SAMPLES:
                    wav = wav[:MAX_SAMPLES]
            else:
                wav = augment_waveform(wav, rng)          # 레거시(v3) 경량 증강
        return {"waveform": wav, "label": int(row["label"])}


def _crop_and_augment(wav: np.ndarray, augment: bool, aug_strength: float) -> np.ndarray:
    """Shared crop (+ optional augmentation) — identical policy to AccentDataset.

    Kept as a module helper so MultiTaskDataset applies the *exact same* window
    crop and augmentation (legacy or domain-randomization) that the validated
    single-task path uses. real and fake clips go through this identically, which
    is the whole point of the multi-task channel-confound control (both sides get
    the same domain randomization so channel can't be a shortcut).
    """
    # AccentDataset.__getitem__ 과 동일한 크롭·증강 정책을 공유 헬퍼로 뺀 것.
    # real·fake 클립이 완전히 같은 변환을 거치게 하여(양쪽 동일 domain_augment) 채널이
    # real/fake 판별의 지름길이 되지 않도록 한다(멀티태스크 채널 confound 통제 핵심).
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
    # 국가 + real/fake 를 함께 학습하기 위한 통합 데이터셋(DATASET.md §11). 통합
    # 매니페스트 컬럼: filename,audio_uri,country,country_label,fake_label,speaker,
    # source,system_id,orig_split. audio_uri 가 이미 완전한 경로이므로(국가 real은
    # curated/<CC>/audio/, ASVspoof 유래/중복복사는 real_fake_5k/audio_asv|audio_dup/)
    # 행마다 자기 경로로 직접 로드한다 — 공유 root 불필요. country_label 은 국가 유래
    # 클립은 0..5, ASVspoof 유래 클립은 -100(국가 손실 무시). fake_label 은 모든
    # 클립에 대해 0=real / 1=fake.
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


@dataclass
class MultiTaskCollator:
    """Normalize + pad a batch and emit *two* label tensors (country + fake).

    Mirrors DataCollator but returns ``country_labels`` and ``fake_labels`` under
    those exact keys so the HF Trainer (with
    ``TrainingArguments.label_names=["country_labels","fake_labels"]``) forwards
    both to the model and back out of ``predict()`` as a label tuple.
    """
    # DataCollator 와 동일하게 정규화·패딩하되, country_labels·fake_labels 두 개의
    # 라벨 텐서를 반환한다. TrainingArguments.label_names 에 이 키들을 등록하면 HF
    # Trainer 가 둘 다 모델로 전달하고 predict() 결과의 label_ids 로도 돌려준다.
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
    # 이미 기록된 train/val/test.csv 매니페스트로부터 세 데이터셋을 만들어 반환.
    train = AccentDataset(Path(manifest_dir) / "train.csv")
    val = AccentDataset(Path(manifest_dir) / "val.csv")
    test = AccentDataset(Path(manifest_dir) / "test.csv")
    return train, val, test
