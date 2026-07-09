import os
import re
import hashlib
import librosa
import numpy as np

from torch.utils.data import Dataset

# LJSpeech 발화 ID (예: LJ001-0001). real 과 그 fake 들이 공유하는 원본 식별자.
_ID_RE = re.compile(r"(LJ\d{3}-\d{4})")

# 계산된 log-mel(1,128,128 float32)을 파일당 한 번만 만들고 .npy 로 캐싱한다.
# 멜 계산은 결정적(랜덤성 없음)이라 캐시는 원본과 완전히 동일 -> epoch 반복 시 디코딩 생략.
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "detector", "mel_cache")
CACHE_DIR = os.path.abspath(CACHE_DIR)


def compute_mel(path):
    """오디오 -> 표준화된 log-mel (1,128,128) float32. 캐시가 담는 값과 동일."""
    audio, sr = librosa.load(path, sr=16000)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel = librosa.power_to_db(mel)
    if mel.shape[1] < 128:
        mel = np.pad(mel, ((0, 0), (0, 128 - mel.shape[1])))
    mel = mel[:, :128].astype(np.float32)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    return np.expand_dims(mel, axis=0)


def _cache_path(path):
    # 심볼릭 링크는 실제 파일 기준으로 캐시(같은 wav 를 여러 곳에서 참조해도 1개 캐시).
    key = hashlib.md5(os.path.realpath(path).encode()).hexdigest()
    return os.path.join(CACHE_DIR, key + ".npy")


def load_mel(path):
    """캐시가 있으면 로드, 없으면 계산 후 원자적으로 저장하고 반환."""
    cp = _cache_path(path)
    if os.path.exists(cp):
        try:
            return np.load(cp)
        except Exception:
            pass  # 손상된 캐시는 재생성
    mel = compute_mel(path)
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = f"{cp}.{os.getpid()}.tmp"  # 원자적 저장(멀티워커 경쟁 대비)
    with open(tmp, "wb") as f:       # 핸들로 저장해야 np.save 가 .npy 를 덧붙이지 않음
        np.save(f, mel)
    os.replace(tmp, cp)
    return mel


def utt_id(path):
    """파일 경로에서 원본 발화 ID를 뽑는다. train/test 를 발화 단위로 가르는 그룹 키.

    같은 발화의 real 과 fake 가 train/test 에 갈리면 모델이 억양이 아니라 문장 내용을
    외워 성능이 뻥튀기되므로, 이 ID 로 묶어서 split 한다.
    """
    m = _ID_RE.search(os.path.basename(path))
    return m.group(1) if m else os.path.basename(path)


class WaveFakeDataset(Dataset):

    def __init__(self, root_dir):

        self.files = []

        real_dir = os.path.join(root_dir, "real")
        fake_dir = os.path.join(root_dir, "fake")

        for f in os.listdir(real_dir):
            self.files.append((os.path.join(real_dir, f), 0))

        for f in os.listdir(fake_dir):
            self.files.append((os.path.join(fake_dir, f), 1))

    @classmethod
    def from_files(cls, files):
        """(path, label) 리스트로 바로 구성. dir 스캔 대신 명시적 split 에 사용."""
        obj = cls.__new__(cls)
        obj.files = list(files)
        return obj

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, label = self.files[idx]
        return load_mel(path), label