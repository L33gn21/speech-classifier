#!/usr/bin/env bash
# Phase 2 (v3 architecture ablation) — proxy sweep over the NEW model levers.
#
# sweep.sh explored the *training* axis (lr x unfreeze). This script explores the
# *architecture* axis added in v3: backbone (wav2vec2 vs WavLM), pooling head
# (mean vs attentive statistics), all-layer weighting, and stronger SpecAugment.
#
# Method (same discipline as v2): hold the proven v2-winner recipe fixed as the
# BASE, change ONE lever per job as a SHORT PROXY (small --per-class, few
# --epochs, early-stopping off so every trial runs the same #steps = fair), then
# read eval_macro_f1 for each. Combine the winning levers into one run and full-
# train that (Phase 3) via submit_job.sh. Ablating one lever at a time keeps the
# comparison clean and cheap; the full 2x2x2 grid is not worth the T4 time.
#
# BASE = v2 winner: full fine-tune (uf12) / lr 3e-5 / balanced / warmup 0.15 / augment.
#
# Tunables (env overrides):
#   PROXY_PER_CLASS=1500   per-class cap for the proxy (smaller = faster)
#   PROXY_EPOCHS=4         epochs per proxy run
#   WAVLM=microsoft/wavlm-base-plus   backbone to try for the a1 ablation
#   ONLY="a0 a1"           run just these ablations (default: all)
#
# Usage:
#   cd gcloud && ./sweep_v3.sh
#   ONLY="a0 a1" ./sweep_v3.sh          # anchor + WavLM only
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/env.sh"

PROXY_PER_CLASS="${PROXY_PER_CLASS:-1500}"
PROXY_EPOCHS="${PROXY_EPOCHS:-4}"
WAVLM="${WAVLM:-microsoft/wavlm-base-plus}"
ONLY="${ONLY:-a0 a1 a2 a3 a4}"

# BASE recipe held fixed across every ablation (the v2 winner). early-stop off so
# each proxy runs the full PROXY_EPOCHS (fair, equal-step comparison).
BASE=( --unfreeze-top=12 --lr=3e-5 --class-weight=balanced --warmup-ratio=0.15
       --augment --per-class="${PROXY_PER_CLASS}" --epochs="${PROXY_EPOCHS}"
       --early-stopping-patience=0 )

# ablation id -> the ONE extra change layered on BASE (label + args)
declare -A LABEL=(
  [a0]="base(w2v/mean/no-lw)"
  [a1]="wavlm-backbone"
  [a2]="attentive-head"
  [a3]="layer-weighting"
  [a4]="strong-specaug"
)
declare -A ARGS=(
  [a0]=""
  [a1]="--backbone=${WAVLM}"
  [a2]="--head=attentive"
  [a3]="--layer-weighting"
  [a4]="--mask-time-prob=0.1"
)

echo ">> v3 architecture ablation sweep"
echo "   base       : ${BASE[*]}"
echo "   proxy      : per-class=${PROXY_PER_CLASS} epochs=${PROXY_EPOCHS} (early-stop off)"
echo "   ablations  : ${ONLY}"
echo "   region=${REGION} bucket=${BUCKET}  (jobs queue if T4 quota < #ablations)"
echo

N=0
for id in ${ONLY}; do
  extra="${ARGS[$id]:-}"
  suffix="v3-${id}-${LABEL[$id]//[^a-zA-Z0-9]/}"
  echo ">> submitting ${id} (${LABEL[$id]}): ${extra:-<no extra>}"
  JOB_SUFFIX="${suffix}" "${HERE}/submit_job.sh" "${BASE[@]}" ${extra}
  N=$((N+1))
  echo
done

echo ">> submitted ${N} proxy ablation jobs."
echo "   track:   gcloud ai custom-jobs list --region=${REGION} --format='table(displayName,state)'"
echo "   compare: read each job's eval_macro_f1 (higher = better):"
echo "     for j in \$(gcloud ai custom-jobs list --region=${REGION} --filter='displayName:accent-classifier-*-v3-*' --format='value(displayName)'); do"
echo "       echo -n \"\$j  \"; gcloud storage cat gs://${BUCKET}/outputs/classifier/\$j/model/final_metrics.json 2>/dev/null | python -c 'import sys,json;d=json.load(sys.stdin);print(\"mf1\",round(d.get(\"eval_macro_f1\",0),3),\"CN\",round(d.get(\"eval_f1_CN\",0),3))'; done"
echo
echo ">> then combine the winning levers and FULL-train, e.g.:"
echo "     JOB_SUFFIX=v3final ./submit_job.sh --auto-register \\"
echo "       --backbone=${WAVLM} --head=attentive --layer-weighting \\"
echo "       --unfreeze-top=12 --lr=3e-5 --class-weight=balanced --warmup-ratio=0.15 \\"
echo "       --per-class=5000 --epochs=10 --early-stopping-patience=3 --augment"
