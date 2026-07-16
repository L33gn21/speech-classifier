#!/usr/bin/env bash
# Deploys the model tester as a separate Cloud Run service. Like the dataset
# dashboard it scales to zero (min-instances=0 -> ~$0 while idle), but it's
# sized bigger because it loads a wav2vec2 model into memory and runs torch
# inference:
#   - memory=4Gi / cpu=2  -> room for the model + torch runtime
#   - max-instances=1     -> single-user tool; also keeps one warm model cache
#   - timeout=300         -> first request for a model downloads weights from GCS
#
# The build context is STAGED into a temp dir so the shared modules from src/
# (config.py, model.py) sit flat next to this dir's files — matching how the
# src/ code imports them (`from config import ...`).
#
# Requires: gcloud/env.sh filled in (PROJECT_ID / REGION / BUCKET), and the
# Cloud Run + Cloud Build APIs enabled.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$(cd "${HERE}/.." && pwd)"
source "${HERE}/../../gcloud/env.sh"

SERVICE="accent-model-tester"

gcloud config set project "${PROJECT_ID}"

echo ">> granting the Cloud Run runtime service account read access to gs://${BUCKET}"
PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
RUNTIME_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member="serviceAccount:${RUNTIME_SA}" \
  --role="roles/storage.objectViewer" >/dev/null

# Optional: link back to the dataset dashboard if it's already deployed.
DATASET_URL="$(gcloud run services describe accent-dataset-dashboard \
  --region "${REGION}" --format='value(status.url)' 2>/dev/null || true)"

# Stage a flat build context: this dir's files + shared src modules.
STAGE="$(mktemp -d)"
trap 'rm -rf "${STAGE}"' EXIT
cp "${HERE}/Dockerfile" "${HERE}/requirements.txt" "${HERE}/serve_model_tester.py" "${STAGE}/"
cp "${SRC}/config.py" "${SRC}/model.py" "${STAGE}/"

echo ">> deploying ${SERVICE} to Cloud Run (${REGION}) — builds via Cloud Build"
gcloud run deploy "${SERVICE}" \
  --source "${STAGE}" \
  --region "${REGION}" \
  --allow-unauthenticated \
  --min-instances=0 \
  --max-instances=1 \
  --memory=4Gi \
  --cpu=2 \
  --concurrency=4 \
  --timeout=300 \
  --set-env-vars="^@^MODEL_ROOT=gs://${BUCKET}/outputs/classifier@DATASET_DASHBOARD_URL=${DATASET_URL}"

echo ">> done. URL:"
gcloud run services describe "${SERVICE}" --region "${REGION}" --format='value(status.url)'
