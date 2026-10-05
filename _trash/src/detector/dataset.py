import os
import re
import hashlib
import librosa
import numpy as np

from torch.utils.data import Dataset

# LJSpeech utterance ID (e.g. LJ001-0001). The source identifier shared by a real clip and its fakes.
_ID_RE = re.compile(r"(LJ\d{3}-\d{4})")

# The computed log-mel (1,128,128 float32) is built once per file and cached as .npy.
# The mel computation is deterministic (no randomness), so the cache equals the original -> decoding is skipped on later epochs.
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "detector", "mel_cache")
CACHE_DIR = os.path.abspath(CACHE_DIR)


def compute_mel(path):
    """Audio -> standardized log-mel (1,128,128) float32. Identical to what the cache stores."""
    audio, sr = librosa.load(path, sr=16000)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel = librosa.power_to_db(mel)
    if mel.shape[1] < 128:
        mel = np.pad(mel, ((0, 0), (0, 128 - mel.shape[1])))
    mel = mel[:, :128].astype(np.float32)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    return np.expand_dims(mel, axis=0)


def _cache_path(path):
    # Symlinks are cached by their real file (one cache even when the same wav is referenced from several places).
    key = hashlib.md5(os.path.realpath(path).encode()).hexdigest()
    return os.path.join(CACHE_DIR, key + ".npy")


def load_mel(path):
    """Load from the cache if present; otherwise compute, save atomically and return."""
    cp = _cache_path(path)
    if os.path.exists(cp):
        try:
            return np.load(cp)
        except Exception:
            pass  # a corrupted cache is regenerated
    mel = compute_mel(path)
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = f"{cp}.{os.getpid()}.tmp"  # atomic save (guards against multi-worker races)
    with open(tmp, "wb") as f:       # save through a handle so np.save does not append .npy
        np.save(f, mel)
    os.replace(tmp, cp)
    return mel


def utt_id(path):
    """Extract the source utterance ID from a file path. The group key that splits train/test by utterance.

    If the real and fake versions of one utterance land on different sides of train/test, the model
    memorizes the sentence content instead of the accent and the score is inflated, so the split is grouped by this ID.
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
        """Build directly from a list of (path, label). Used for explicit splits instead of a dir scan."""
        obj = cls.__new__(cls)
        obj.files = list(files)
        return obj

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, label = self.files[idx]
        return load_mel(path), label