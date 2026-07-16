# Vertex AI Custom Training container for the accent classifier.
# Base image bundles a CUDA-enabled torch; we add a matching torchaudio + deps.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

# ffmpeg + libsndfile: mp3 decoding for torchaudio/librosa/soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torchaudio==2.4.1 \
        --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir -r requirements.txt

# training code (flat layout: config/model/dataset/... import each other directly)
COPY src/*.py ./

ENV PYTHONUNBUFFERED=1
# Vertex passes extra --args here (e.g. --epochs=8 --unfreeze-top=4).
ENTRYPOINT ["python", "train.py"]
