# Cloud Run image for the model tester. Separate from the dataset dashboard
# image (../dashboard/Dockerfile) because this one carries torch + transformers
# + ffmpeg to run inference server-side.
#
# The build context is staged by deploy.sh so that the shared modules
# (config.py, model.py) sit next to this dir's files at the context root —
# imports are flat (`from config import ...`), matching the src/ layout.
FROM python:3.11-slim

WORKDIR /app

# ffmpeg: decodes any uploaded/recorded container (webm/opus from the browser,
# mp3/wav/m4a from uploads) to raw PCM for the model.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch keeps the image a few GB smaller than the CUDA build. Version
# matches training (classifier/requirements.txt).
RUN pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the wav2vec2-base config (small JSON) so building the backbone skeleton
# at inference time needs no network to HuggingFace. Weights come from GCS.
RUN python -c "from transformers import Wav2Vec2Config; Wav2Vec2Config.from_pretrained('facebook/wav2vec2-base')"

# Shared modules + this service (staged flat by deploy.sh).
COPY config.py model.py serve_model_tester.py ./

# 1 worker: the model cache lives in one process's memory, so a single warm
# worker maximizes cache hits. Threads handle the light concurrent traffic.
# Long timeout: the first request for a model downloads weights from GCS.
CMD exec gunicorn --bind :${PORT:-8080} --workers 1 --threads 4 --timeout 300 serve_model_tester:app
