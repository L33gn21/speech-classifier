import sys
from pathlib import Path

import torch
import librosa
import numpy as np

from model import Detector

# inference.py lives at src/detector/inference.py -> project root is three levels up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WEIGHTS = PROJECT_ROOT / "outputs" / "detector" / "detector.pt"

# usage: python src/detector/inference.py <audio.wav>  (defaults to sample.wav)
audio_path = sys.argv[1] if len(sys.argv) > 1 else "sample.wav"

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Detector().to(device)
model.load_state_dict(torch.load(WEIGHTS, map_location=device))
model.eval()

audio, sr = librosa.load(
    audio_path,
    sr=16000
)

mel = librosa.feature.melspectrogram(
    y=audio,
    sr=sr,
    n_mels=128
)

mel = librosa.power_to_db(mel)

if mel.shape[1] < 128:
    mel = np.pad(mel, ((0,0),(0,128-mel.shape[1])))

mel = mel[:, :128]

mel = (mel - mel.mean()) / (mel.std() + 1e-6)

mel = torch.tensor(mel).unsqueeze(0).unsqueeze(0).float().to(device)

with torch.no_grad():

    pred = model(mel)

    label = pred.argmax(1).item()

print("Fake" if label else "Real")