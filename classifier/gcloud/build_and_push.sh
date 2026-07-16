#!/usr/bin/env bash
# Phase 2 -> container. Builds the training image with Cloud Build (no local
# Docker needed) and pushes it to Artifact Registry. Creates the repo if absent.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/env.sh"
BUILD_CONTEXT="$(cd "${HERE}/.." && pwd)"   # classifier/ (holds the Dockerfile)

gcloud config set project "${PROJECT_ID}"

# ensure the Artifact Registry docker repo exists
if ! gcloud artifacts repositories describe "${REPO}" --location="${REGION}" >/dev/null 2>&1; then
  echo ">> creating Artifact Registry repo ${REPO} in ${REGION}"
  gcloud artifacts repositories create "${REPO}" \
    --repository-format=docker --location="${REGION}"
fi

echo ">> building & pushing ${IMAGE_URI}"
gcloud builds submit --tag "${IMAGE_URI}" "${BUILD_CONTEXT}"

echo "done: ${IMAGE_URI}"
