#!/usr/bin/env bash
# Deploys the model tester as a separate Cloud Run service. It's sized for a
# torch model in memory, not a static page:
#   - memory=4Gi / cpu=2  -> room for the model + torch runtime
#   - max-instances=1     -> demo-scale traffic only; also keeps one warm model cache
#   - min-instances=1     -> always-on for the /api/* consumers (other servers
#                            poll this continuously) -> no scale-to-zero, so this
#                            service now costs ~24/7 instead of ~$0 while idle
#   - timeout=300         -> first request for a model downloads weights from GCS
#
# Auth: the browser UI (/,/predict,...) still uses the session login above.
# The machine-facing /api/* routes use a static X-API-Key header instead
# (see API_KEY below, and api_key_required in serve_model_tester.py) — demo-grade,
# not a real secrets-managed key, rotate it by re-running this script with a new
# API_KEY env var.
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

# API_KEY: set it in your shell before running this script to pin a real value,
# e.g. `API_KEY=$(openssl rand -hex 16) ./deploy.sh`. Otherwise falls back to
# the same insecure default baked into serve_model_tester.py — fine for a demo,
# not for anything you'd call "production".
API_KEY="${API_KEY:-dev-key-change-me}"

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
  --min-instances=1 \
  --max-instances=1 \
  --memory=4Gi \
  --cpu=2 \
  --concurrency=4 \
  --timeout=300 \
  --set-env-vars="^@^MODEL_ROOT=gs://${BUCKET}/outputs/classifier@DATASET_DASHBOARD_URL=${DATASET_URL}@API_KEY=${API_KEY}"

echo ">> done. URL:"
gcloud run services describe "${SERVICE}" --region "${REGION}" --format='value(status.url)'
