#!/usr/bin/env bash
# Channel-leakage probe (DATASET.md §5.1) -> Vertex AI CPU Custom Job.
#
# Runs src/channel_probe.py in the pushed container to measure how much of the
# classifier's separability comes from the recording channel rather than accent
# (the leading suspect for the v3 CA->US collapse on VoxForge). It trains simple
# linear probes on low-level / silence-only acoustic features over the SAME
# speaker-disjoint splits the model uses, and writes channel_probe_report.json.
#
# CPU-ONLY on purpose: the probe is librosa + scikit-learn (no torch/GPU), so we
# force no accelerator. Bonus: it sidesteps the us-west2 T4 queue waits entirely.
#
# Prereq: rebuild the image first so the new channel_probe.py is baked in:
#   cd gcloud && ./build_and_push.sh
#
# Any args after the script are forwarded to channel_probe.py, e.g.:
#   ./submit_probe_job.sh --per-class=1500
#   ./submit_probe_job.sh --per-class=800 --max-seconds=6
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/env.sh"

gcloud config set project "${PROJECT_ID}"

JOB_NAME="channel-probe-$(date +%Y%m%d-%H%M%S)${JOB_SUFFIX:+-${JOB_SUFFIX}}"
CURATED_ROOT="gs://${BUCKET}/curated"
OUTPUT_BASE="gs://${BUCKET}/outputs/classifier/${JOB_NAME}"

# default probe args if none passed
if [ "$#" -eq 0 ]; then
  set -- --per-class=1500
fi

# build the container args block (YAML list) from "$@"
ARGS_YAML=""
for a in "$@"; do
  ARGS_YAML+="          - \"${a}\""$'\n'
done

# No accelerator block: the probe is CPU-only. MACHINE_TYPE (env, default
# n1-standard-8) gives ~8 vCPUs which channel_probe.py uses to featurize clips in
# parallel (--n-jobs=-1). We override the image ENTRYPOINT (python train.py) with
# a command that runs the probe instead.
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
        - "channel_probe.py"
      env:
        - name: CV_CURATED_ROOT
          value: ${CURATED_ROOT}
      args:
${ARGS_YAML}
YAML

echo ">> submitting ${JOB_NAME} (CPU, no accelerator)"
echo "   image  : ${IMAGE_URI}"
echo "   data   : ${CURATED_ROOT}"
echo "   report : ${OUTPUT_BASE}/model/channel_probe_report.json  (AIP_MODEL_DIR)"
echo "   args   :" "$@"

JOB_ID="$(gcloud ai custom-jobs create \
  --region="${REGION}" \
  --display-name="${JOB_NAME}" \
  --config="${CONFIG}" \
  --format="value(name)")"

rm -f "${CONFIG}"
echo ">> submitted: ${JOB_ID}"
echo "   track:  gcloud ai custom-jobs list --region=${REGION} --format='table(displayName,state)'"
echo "   result (when JOB_STATE_SUCCEEDED):"
echo "     gcloud storage cat ${OUTPUT_BASE}/model/channel_probe_report.json | python -c 'import sys,json;print(json.dumps(json.load(sys.stdin)[\"verdict\"],indent=2))'"
