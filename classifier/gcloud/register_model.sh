#!/usr/bin/env bash
# Register a trained model artifact (already in GCS) into the Vertex AI Model
# Registry — CATALOG / VERSIONING ONLY.
#
# What this does: takes the model dir a Custom Job wrote to
#   gs://<BUCKET>/outputs/classifier/<JOB>/model/
# and registers it as a versioned entry in the regional Model Registry, so you
# get version history, lineage and metadata pointing back at the GCS artifacts.
#
# What this does NOT do: make the model deployable to an Endpoint. This is a
# custom wav2vec2 model — the saved artifacts are raw weights that only load via
# src/model.py's AccentClassifier (+ the legacy weight_norm shim, see infer.py).
# A stock prebuilt serving container CANNOT serve it. The --container-image-uri
# below is therefore a NOMINAL placeholder required by `models upload`; it is not
# run at registration time. To actually serve, build a custom serving container
# wrapping infer.py (/health + /predict) and register THAT (see docs).
#
# 학습 산출물(GCS)을 Vertex AI Model Registry에 "카탈로그/버전관리 목적으로만"
# 등록한다. 버전 이력·계보·메타데이터가 GCS 아티팩트를 가리키게 된다.
# 이 커스텀 wav2vec2 모델은 prebuilt 서빙 컨테이너로 배포 불가라서, 아래
# 컨테이너 URI는 upload가 요구하는 "형식상 placeholder"일 뿐 등록 시 실행되지
# 않는다. 실제 배포는 infer.py를 감싼 커스텀 서빙 컨테이너가 필요하다.
#
# Usage:
#   ./register_model.sh <JOB_NAME>                 # registers gs://<BUCKET>/outputs/classifier/<JOB_NAME>/model
#   ./register_model.sh gs://bucket/path/to/model  # or an explicit model dir
#
# Examples:
#   ./register_model.sh accent-classifier-20260716-120000
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/env.sh"

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <JOB_NAME | gs://.../model>" >&2
  exit 2
fi

# Resolve the artifact URI: a bare job name expands under the standard outputs path.
ARG="$1"
if [[ "${ARG}" == gs://* ]]; then
  ARTIFACT_URI="${ARG%/}"
else
  ARTIFACT_URI="gs://${BUCKET}/outputs/classifier/${ARG}/model"
fi

# Model Registry display name (stable, groups versions). Override via env if desired.
MODEL_DISPLAY_NAME="${MODEL_DISPLAY_NAME:-accent-classifier}"

# Nominal serving container (NOT capable of serving this custom model — see header).
# Required field for `models upload` but never run at registration time. We default
# to THIS PROJECT'S OWN training image: a prebuilt PyTorch prediction image makes
# `models upload` validate the artifact for TorchServe (demands model.mar) and
# fail on our raw wav2vec2 weights; a custom image URI skips that validation.
SERVING_IMAGE="${SERVING_IMAGE:-${IMAGE_URI}}"

gcloud config set project "${PROJECT_ID}" >/dev/null

# Preflight: the artifact dir must actually contain weights.
# Use `gcloud storage` (not `gsutil stat`) — this project standardizes on the
# gcloud storage CLI, and `gsutil` isn't reliably available/authed here, which
# made this check spuriously fail and abort --auto-register.
if ! gcloud storage ls "${ARTIFACT_URI}/model.safetensors" >/dev/null 2>&1 \
   && ! gcloud storage ls "${ARTIFACT_URI}/pytorch_model.bin" >/dev/null 2>&1; then
  echo ">> ERROR: no model.safetensors / pytorch_model.bin under ${ARTIFACT_URI}" >&2
  echo "   (is the JOB_NAME correct and the training job finished?)" >&2
  exit 1
fi

# Versioning: if a model with this display name already exists, add a new version
# under it (--parent-model). Otherwise create the first version.
EXISTING_ID="$(gcloud ai models list \
  --region="${REGION}" \
  --filter="displayName=${MODEL_DISPLAY_NAME}" \
  --format="value(name)" 2>/dev/null | head -n1 || true)"

echo ">> registering into Model Registry (catalog/versioning only)"
echo "   region     : ${REGION}"
echo "   display    : ${MODEL_DISPLAY_NAME}"
echo "   artifact   : ${ARTIFACT_URI}"
echo "   serving img: ${SERVING_IMAGE}  (nominal — not deployable as-is)"

# Common flags. --version-description records where this version came from.
COMMON_ARGS=(
  --region="${REGION}"
  --display-name="${MODEL_DISPLAY_NAME}"
  --artifact-uri="${ARTIFACT_URI}"
  --container-image-uri="${SERVING_IMAGE}"
  --version-description="artifact: ${ARTIFACT_URI}"
)

if [ -n "${EXISTING_ID}" ]; then
  # --parent-model needs the FULL resource name. `models list` returns only the
  # numeric id, so rebuild the path — passing the bare id fails with
  # "parent_model: Location ID is not provided".
  PARENT="projects/${PROJECT_ID}/locations/${REGION}/models/${EXISTING_ID##*/}"
  echo "   -> existing model found; adding a new version under ${PARENT}"
  gcloud ai models upload "${COMMON_ARGS[@]}" --parent-model="${PARENT}"
else
  echo "   -> no existing model; creating the first version"
  gcloud ai models upload "${COMMON_ARGS[@]}"
fi

echo ">> done. Inspect it:"
echo "   gcloud ai models list --region=${REGION} --filter=\"displayName=${MODEL_DISPLAY_NAME}\""
echo "   console: https://console.cloud.google.com/vertex-ai/models?project=${PROJECT_ID}"
