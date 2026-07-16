#!/usr/bin/env bash
# Phase 3 -> Vertex AI. Submits a Custom Training job that runs train.py in the
# pushed container. Data is read from GCS (FUSE-mounted at /gcs); the model is
# written to AIP_MODEL_DIR under baseOutputDirectory.
#
# Any args after the script are forwarded to train.py, e.g.:
#   ./submit_job.sh --epochs=8 --batch-size=8 --grad-accum=2
#   ./submit_job.sh --unfreeze-top=4 --lr=2e-5 --epochs=6
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/env.sh"

gcloud config set project "${PROJECT_ID}"

JOB_NAME="accent-classifier-$(date +%Y%m%d-%H%M%S)"
# curated 풀(DATASET.md)을 데이터 소스로 사용. 컨테이너 안에서 train.py 가
# 이 매니페스트들을 읽어 화자 단위 train/val/test 분할을 직접 만든다.
CURATED_ROOT="gs://${BUCKET}/curated"
OUTPUT_BASE="gs://${BUCKET}/outputs/classifier/${JOB_NAME}"

# default train.py args if none passed
if [ "$#" -eq 0 ]; then
  set -- --epochs=4 --batch-size=8 --grad-accum=2
fi

# build the container args block (YAML list) from "$@"
ARGS_YAML=""
for a in "$@"; do
  ARGS_YAML+="          - \"${a}\""$'\n'
done

# Accelerator is optional: set ACCELERATOR_TYPE=none (or empty) in env.sh to run
# on CPU only (no Vertex GPU quota needed). Otherwise the T4/L4 block is added.
# 가속기는 선택 사항이다. env.sh 에서 ACCELERATOR_TYPE=none(또는 빈 값)으로
# 두면 GPU 쿼터 없이 CPU만으로 실행한다. 그 외에는 T4/L4 블록이 추가된다.
ACCEL_YAML=""
if [ -n "${ACCELERATOR_TYPE:-}" ] && [ "${ACCELERATOR_TYPE}" != "none" ]; then
  ACCEL_YAML="      acceleratorType: ${ACCELERATOR_TYPE}"$'\n'"      acceleratorCount: ${ACCELERATOR_COUNT}"$'\n'
fi

# TensorBoard streaming is optional: set TENSORBOARD_ID + SERVICE_ACCOUNT in
# env.sh to have Vertex AI live-sync loss/accuracy/F1 curves to a TensorBoard
# instance as the job runs. Both fields require each other (a service account
# is needed for Vertex to write into the TensorBoard resource on your behalf).
TB_YAML=""
if [ -n "${TENSORBOARD_ID:-}" ] && [ -n "${SERVICE_ACCOUNT:-}" ]; then
  TB_YAML="tensorboard: ${TENSORBOARD_ID}"$'\n'"serviceAccount: ${SERVICE_ACCOUNT}"$'\n'
fi

CONFIG="$(mktemp)"
cat > "${CONFIG}" <<YAML
baseOutputDirectory:
  outputUriPrefix: ${OUTPUT_BASE}
${TB_YAML}workerPoolSpecs:
  - machineSpec:
      machineType: ${MACHINE_TYPE}
${ACCEL_YAML}    replicaCount: 1
    containerSpec:
      imageUri: ${IMAGE_URI}
      env:
        - name: CV_CURATED_ROOT
          value: ${CURATED_ROOT}
      args:
${ARGS_YAML}
YAML

echo ">> submitting ${JOB_NAME}"
echo "   image      : ${IMAGE_URI}"
echo "   data       : ${CURATED_ROOT}"
echo "   output     : ${OUTPUT_BASE}/model  (AIP_MODEL_DIR)"
if [ -n "${TB_YAML}" ]; then
  echo "   tensorboard: ${TENSORBOARD_ID}"
fi
echo "   args       :" "$@"

gcloud ai custom-jobs create \
  --region="${REGION}" \
  --display-name="${JOB_NAME}" \
  --config="${CONFIG}"

rm -f "${CONFIG}"
echo ">> submitted. Track it:"
echo "   gcloud ai custom-jobs list --region=${REGION}"
