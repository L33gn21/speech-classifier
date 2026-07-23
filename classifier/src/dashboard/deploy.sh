#!/usr/bin/env bash
# Deploys the dataset dashboard as an always-on Cloud Run service, tuned for
# minimum cost:
#   - min-instances=0  -> scales to zero when idle, $0 while nobody's looking
#   - max-instances=1  -> caps the worst case, this is a single-user tool
#   - 512Mi / 1 vCPU   -> smallest practical size for pandas+matplotlib
#   - Cloud Run's free tier (2M requests, 360k GB-s, 180k vCPU-s per month)
#     covers this workload's traffic pattern easily -> expect ~$0/month.
#
# Requires: gcloud/env.sh filled in (PROJECT_ID / REGION / BUCKET), and the
# Cloud Run + Cloud Build APIs enabled (run.googleapis.com,
# cloudbuild.googleapis.com — cloudbuild is already required by
# build_and_push.sh for the training image).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/../../gcloud/env.sh"

SERVICE="accent-dataset-dashboard"

gcloud config set project "${PROJECT_ID}"

echo ">> granting the Cloud Run runtime service account read access to gs://${BUCKET}"
PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
RUNTIME_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member="serviceAccount:${RUNTIME_SA}" \
  --role="roles/storage.objectViewer" >/dev/null

echo ">> deploying ${SERVICE} to Cloud Run (${REGION}) — builds via Cloud Build from this dir"
gcloud run deploy "${SERVICE}" \
  --source "${HERE}" \
  --region "${REGION}" \
  --allow-unauthenticated \
  --min-instances=0 \
  --max-instances=1 \
  --memory=512Mi \
  --cpu=1 \
  --concurrency=80 \
  --timeout=120 \
  --set-env-vars="^@^DATASET_ROOT=gs://${BUCKET}/curated@DATASET_CLASSES=US,UK,IN,NG,CA,JP,CN,AU,KR@SPOOF_ROOT=gs://${BUCKET}/curated_spoof/asvspoof2019_la"

echo ">> done. URL:"
gcloud run services describe "${SERVICE}" --region "${REGION}" --format='value(status.url)'
