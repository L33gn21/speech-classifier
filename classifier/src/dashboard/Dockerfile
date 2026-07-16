# Cloud Run image for the dataset dashboard. Deliberately separate from the
# training Dockerfile (../../Dockerfile) — no CUDA/torch, so this is small
# (fast cold starts, cheap to store/run).
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY inspect_dataset.py serve_dataset_report.py ./

# Cloud Run sets $PORT (defaults to 8080); gunicorn binds to it directly.
# 1 worker, 8 threads: this is an I/O-bound (GCS) + light-CPU (matplotlib)
# dashboard with very low concurrent traffic, so no need for more.
CMD exec gunicorn --bind :${PORT:-8080} --workers 1 --threads 8 --timeout 120 serve_dataset_report:app
