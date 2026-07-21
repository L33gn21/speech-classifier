#!/usr/bin/env bash
# Evaluate a trained model on an arbitrary curated-root -> Vertex AI CPU Custom Job.
#
# Overrides the container ENTRYPOINT (python train.py) with `python evaluate.py`
# and forwards --model-dir / --curated-root / --manifest-dir. CPU-ONLY on purpose
# (inference over a small set; no T4 queue wait). Reads model + eval data from GCS
# via the /gcs FUSE mount, and writes <model-dir>/test_report.json there.
#
# Used to score a model on the unseen VoxForge staging set (gcloud-first; the v3
# run did this on a local CPU which DATASET/CLAUDE flagged as second-best).
#
# Usage (VoxForge, 750 clips, 5 classes):
#   MODEL=/gcs/qi-ucsd-speech-usw2/outputs/classifier/<JOB>/model \
#   CURATED=/gcs/qi-ucsd-speech-usc1/test_voxforge \
#   MANIFEST=/gcs/qi-ucsd-speech-usc1/test_voxforge/manifests \
#   JOB_SUFFIX=voxforge-<JOB> ./submit_eval_job.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/env.sh"

: "${MODEL:?set MODEL=/gcs/<bucket>/.../model}"
: "${CURATED:?set CURATED=/gcs/<bucket>/<eval-root>}"
MANIFEST="${MANIFEST:-${CURATED}/manifests}"

gcloud config set project "${PROJECT_ID}"

JOB_NAME="eval-$(date +%Y%m%d-%H%M%S)${JOB_SUFFIX:+-${JOB_SUFFIX}}"
# Output base is only used for Vertex bookkeeping; evaluate.py writes its report
# into --model-dir (the model's own dir), so the comparison artifact sits with it.
OUTPUT_BASE="gs://${BUCKET}/outputs/classifier/${JOB_NAME}"

CONFIG="$(mktemp)"
cat > "${CONFIG}" <<YAML
baseOutputDirectory:
  outputUriPrefix: ${OUTPUT_BASE}
workerPoolSpecs:
  - machineSpec:
      machineType: ${MACHINE_TYPE}
    replicaCount: 1
    containerSpec:
      imageUri: ${IMAGE_URI}
      command:
        - "python"
        - "evaluate.py"
      args:
        - "--model-dir=${MODEL}"
        - "--curated-root=${CURATED}"
        - "--manifest-dir=${MANIFEST}"
YAML

echo ">> submitting ${JOB_NAME} (CPU, no accelerator)"
echo "   image    : ${IMAGE_URI}"
echo "   model    : ${MODEL}"
echo "   eval data: ${CURATED}  (manifests: ${MANIFEST})"
echo "   report   : ${MODEL}/test_report.json"

JOB_ID="$(gcloud ai custom-jobs create \
  --region="${REGION}" \
  --display-name="${JOB_NAME}" \
  --config="${CONFIG}" \
  --format="value(name)")"

rm -f "${CONFIG}"
echo ">> submitted: ${JOB_ID}"
echo "   track: gcloud ai custom-jobs list --region=${REGION} --format='table(displayName,state)'"
