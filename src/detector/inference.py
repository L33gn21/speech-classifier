import torch
import librosa
import numpy as np

from model import Detector

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Detector().to(device)
model.load_state_dict(torch.load("detector.pt"))
model.eval()

audio, sr = librosa.load(
    "sample.wav",
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