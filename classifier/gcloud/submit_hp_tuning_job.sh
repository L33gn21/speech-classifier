#!/usr/bin/env bash
# Phase 2 (proper path, OPTIONAL) — Vertex AI Hyperparameter Tuning (Vizier).
#
# Runs many trials of the SAME container, letting Vizier (Bayesian black-box
# optimizer) pick each trial's lr / unfreeze_top / weight_decay from previous
# results. Requires the --hypertune wiring in train.py (reports eval_macro_f1
# via cloudml-hypertune) — already added. See docs/hyperparameter-tuning.md.
#
# COST WARNING (CLAUDE.md §3): total ≈ maxTrialCount x one-training-cost. With a
# single-T4 quota, parallelTrialCount is effectively 1 and trials run serially,
# so wall-time ≈ maxTrialCount x per-trial. Prefer ./sweep.sh (a fixed small
# grid) unless you specifically want Vizier's search. Report cost before running.
#
# Env overrides:
#   MAX_TRIALS=12  PARALLEL_TRIALS=1  PROXY_PER_CLASS=1500  PROXY_EPOCHS=3
#
# Usage:  cd gcloud && ./submit_hp_tuning_job.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/env.sh"

MAX_TRIALS="${MAX_TRIALS:-12}"
PARALLEL_TRIALS="${PARALLEL_TRIALS:-1}"   # keep <= T4 quota or trials queue
PROXY_PER_CLASS="${PROXY_PER_CLASS:-1500}"
PROXY_EPOCHS="${PROXY_EPOCHS:-3}"

gcloud config set project "${PROJECT_ID}"

JOB_NAME="accent-hp-$(date +%Y%m%d-%H%M%S)"
CURATED_ROOT="gs://${BUCKET}/curated"
OUTPUT_BASE="gs://${BUCKET}/outputs/classifier/${JOB_NAME}"

ACCEL_YAML=""
if [ -n "${ACCELERATOR_TYPE:-}" ] && [ "${ACCELERATOR_TYPE}" != "none" ]; then
  ACCEL_YAML="        acceleratorType: ${ACCELERATOR_TYPE}"$'\n'"        acceleratorCount: ${ACCELERATOR_COUNT}"$'\n'
fi

# Fixed args every trial shares (the proxy budget + recipe). Vizier APPENDS the
# tuned params as --lr=.. --unfreeze_top=.. --weight_decay=.. (parameterId with
# underscores; train.py has matching aliases). --hypertune makes train.py report
# eval_macro_f1 back to Vizier as the trial score.
CONFIG="$(mktemp)"
cat > "${CONFIG}" <<YAML
displayName: ${JOB_NAME}
studySpec:
  algorithm: ALGORITHM_UNSPECIFIED
  metrics:
    - metricId: macro_f1
      goal: MAXIMIZE
  parameters:
    - parameterId: lr
      scaleType: UNIT_LOG_SCALE
      doubleValueSpec:
        minValue: 0.00001
        maxValue: 0.0003
    - parameterId: unfreeze_top
      discreteValueSpec:
        values: [0, 2, 4, 6]
    - parameterId: weight_decay
      doubleValueSpec:
        minValue: 0.0
        maxValue: 0.1
maxTrialCount: ${MAX_TRIALS}
parallelTrialCount: ${PARALLEL_TRIALS}
trialJobSpec:
  baseOutputDirectory:
    outputUriPrefix: ${OUTPUT_BASE}
  workerPoolSpecs:
    - machineSpec:
        machineType: ${MACHINE_TYPE}
${ACCEL_YAML}      replicaCount: 1
      containerSpec:
        imageUri: ${IMAGE_URI}
        env:
          - name: CV_CURATED_ROOT
            value: ${CURATED_ROOT}
        args:
          - "--hypertune"
          - "--class-weight=balanced"
          - "--augment"
          - "--early-stopping-patience=0"
          - "--per-class=${PROXY_PER_CLASS}"
          - "--epochs=${PROXY_EPOCHS}"
YAML

echo ">> submitting HP tuning job ${JOB_NAME}"
echo "   trials     : max=${MAX_TRIALS} parallel=${PARALLEL_TRIALS} (proxy: per-class=${PROXY_PER_CLASS} epochs=${PROXY_EPOCHS})"
echo "   image      : ${IMAGE_URI}"
echo "   data       : ${CURATED_ROOT}"
echo "   output     : ${OUTPUT_BASE}"

gcloud ai hp-tuning-jobs create \
  --region="${REGION}" \
  --display-name="${JOB_NAME}" \
  --config="${CONFIG}"

rm -f "${CONFIG}"
echo ">> submitted. track: gcloud ai hp-tuning-jobs list --region=${REGION}"
