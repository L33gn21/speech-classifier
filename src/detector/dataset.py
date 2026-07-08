import os
import librosa
import numpy as np

from torch.utils.data import Dataset

class WaveFakeDataset(Dataset):

    def __init__(self, root_dir):

        self.files = []

        real_dir = os.path.join(root_dir, "real")
        fake_dir = os.path.join(root_dir, "fake")

        for f in os.listdir(real_dir):
            self.files.append((os.path.join(real_dir, f), 0))

        for f in os.listdir(fake_dir):
            self.files.append((os.path.join(fake_dir, f), 1))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        path, label = self.files[idx]

        audio, sr = librosa.load(
            path,
            sr=16000
        )

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=128
        )

        mel = librosa.power_to_db(mel)

        if mel.shape[1] < 128:
            pad = 128 - mel.shape[1]
            mel = np.pad(mel, ((0,0),(0,pad)))

        mel = mel[:, :128]

        mel = mel.astype(np.float32)

        mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        mel = np.expand_dims(mel, axis=0)

        return mel, label