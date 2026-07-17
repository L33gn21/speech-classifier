#!/usr/bin/env bash
# Phase 2 (light path) — a small manual hyperparameter grid.
#
# Submits one Vertex Custom Job per (lr x unfreeze-top) combo as SHORT PROXY runs
# (reduced --per-class + few --epochs) so each finishes fast and cheap. With a
# single-T4 quota the jobs queue and run sequentially; that's fine. After they
# finish, pick the best combo by eval_macro_f1 (see the printed compare command),
# then launch ONE full run with that combo (Phase 3) via submit_job.sh.
#
# Why the light path (vs Vertex Vizier / submit_hp_tuning_job.sh): with only ~1
# parallel T4, a Bayesian sweep runs serially and costs trial_count x train_time.
# A 4-point grid is enough to see the lr/unfreeze landscape at a fraction of that.
#
# Tunables (env overrides):
#   LRS="5e-5 1e-4"        learning rates to try
#   UNFREEZE="2 4"          top-N transformer layers to unfreeze
#   PROXY_PER_CLASS=1500    per-class cap for the proxy (smaller = faster)
#   PROXY_EPOCHS=3          epochs per proxy run
#   EXTRA="--class-weight balanced --augment"   forwarded to every run
#
# Usage:
#   cd gcloud && ./sweep.sh
#   LRS="3e-5 1e-4 2e-4" UNFREEZE="4 6" ./sweep.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/env.sh"

LRS="${LRS:-5e-5 1e-4}"
UNFREEZE="${UNFREEZE:-2 4}"
PROXY_PER_CLASS="${PROXY_PER_CLASS:-1500}"
PROXY_EPOCHS="${PROXY_EPOCHS:-3}"
# Early stopping off for the proxy so every trial runs the same #epochs (fair
# comparison); class weighting + augmentation on to match the intended recipe.
EXTRA="${EXTRA:---class-weight balanced --augment}"

echo ">> sweep: lrs=[${LRS}] unfreeze=[${UNFREEZE}] per-class=${PROXY_PER_CLASS} epochs=${PROXY_EPOCHS}"
echo "   region=${REGION} bucket=${BUCKET}  (jobs queue if T4 quota < grid size)"
echo "   extra=${EXTRA}"
echo

N=0
for lr in ${LRS}; do
  for uf in ${UNFREEZE}; do
    # compact, filesystem-safe suffix for the job name (e.g. lr5e-5 -> lr5e5)
    lrtag="lr$(echo "${lr}" | tr -d '.-' )"
    suffix="${lrtag}-uf${uf}"
    echo ">> submitting ${suffix}: --lr=${lr} --unfreeze-top=${uf}"
    JOB_SUFFIX="${suffix}" "${HERE}/submit_job.sh" \
      --lr="${lr}" --unfreeze-top="${uf}" \
      --per-class="${PROXY_PER_CLASS}" --epochs="${PROXY_EPOCHS}" \
      --early-stopping-patience=0 ${EXTRA}
    N=$((N+1))
    echo
  done
done

echo ">> submitted ${N} proxy jobs."
echo "   track:   gcloud ai custom-jobs list --region=${REGION} --format='table(displayName,state)'"
echo "   compare: for each finished job, read its eval_macro_f1:"
echo "     for j in \$(gcloud ai custom-jobs list --region=${REGION} --filter='displayName:accent-classifier-*-uf*' --format='value(displayName)'); do"
echo "       echo -n \"\$j  \"; gcloud storage cat gs://${BUCKET}/outputs/classifier/\$j/model/final_metrics.json 2>/dev/null | python -c 'import sys,json;print(json.load(sys.stdin).get(\"eval_macro_f1\"))'; done"
