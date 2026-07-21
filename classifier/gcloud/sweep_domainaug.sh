#!/usr/bin/env bash
# Phase 2 (v4 domain-adaptation) — proxy sweep over DOMAIN-RANDOMIZATION strength.
#
# Context: v3 (WavLM/mean) generalizes poorly to the unseen VoxForge corpus,
# where CA collapses into US (F1 .57 -> .32). Three diagnostics converged on the
# cause: it is NOT channel leakage but real accent similarity + DOMAIN SHIFT
# (GLOBE clean-24kHz TTS-grade -> VoxForge amateur home recordings). The tuning
# axis is exhausted (HP / attentive head / strong SpecAugment all regressed on
# VoxForge). This sweep opens the remaining code-side lever: (C) domain
# adaptation via realistic recording-condition augmentation (speed / band-limit /
# reverb / colored-noise), exposed as train.py --aug-strength.
#
# Method (same discipline as sweep_v3.sh): hold the v3 winner recipe fixed as the
# BASE, change ONLY --aug-strength per job as a SHORT PROXY (small --per-class,
# few --epochs, early-stopping off so every trial runs the same #steps = fair),
# read eval_macro_f1. The proxy picks a promising strength, but — per the
# country6 sweep lesson (attentive looked good on the proxy, then regressed on
# VoxForge) — the WINNER MUST BE CONFIRMED with a full train + VoxForge eval
# (submit_eval_job.sh) before we trust it. Proxy signal alone is not acceptance.
#
# BASE = v3 winner: WavLM / mean / uf12 / lr 3e-5 / balanced / warmup 0.15 / augment.
#   d0 (anchor) = --aug-strength 0  == legacy light aug == the v3 recipe (control).
#
# Tunables (env overrides):
#   PROXY_PER_CLASS=1500   per-class cap for the proxy (smaller = faster)
#   PROXY_EPOCHS=4         epochs per proxy run
#   WAVLM=microsoft/wavlm-base-plus
#   ONLY="d0 d1 d2 d3"     run just these (default: all)
#
# Usage:
#   cd gcloud && ./build_and_push.sh     # domain_augment is new code — rebuild first!
#   cd gcloud && ./sweep_domainaug.sh
#   ONLY="d0 d2" ./sweep_domainaug.sh    # control + strength 1.0 only
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/env.sh"

PROXY_PER_CLASS="${PROXY_PER_CLASS:-1500}"
PROXY_EPOCHS="${PROXY_EPOCHS:-4}"
WAVLM="${WAVLM:-microsoft/wavlm-base-plus}"
ONLY="${ONLY:-d0 d1 d2 d3}"

# BASE recipe held fixed across every trial (the v3 winner). early-stop off so
# each proxy runs the full PROXY_EPOCHS (fair, equal-step comparison). --augment
# is always on; only --aug-strength changes (0 = legacy light aug = v3 control).
BASE=( --backbone="${WAVLM}" --unfreeze-top=12 --lr=3e-5 --class-weight=balanced
       --warmup-ratio=0.15 --augment --per-class="${PROXY_PER_CLASS}"
       --epochs="${PROXY_EPOCHS}" --early-stopping-patience=0 )

# trial id -> the ONE lever changed on BASE (label + the --aug-strength value)
declare -A LABEL=(
  [d0]="anchor-legacyaug(v3)"
  [d1]="domainaug-0.5"
  [d2]="domainaug-1.0"
  [d3]="domainaug-1.5"
)
declare -A STRENGTH=(
  [d0]="0.0"
  [d1]="0.5"
  [d2]="1.0"
  [d3]="1.5"
)

echo ">> v4 domain-randomization strength sweep (WavLM base)"
echo "   base       : ${BASE[*]}"
echo "   proxy      : per-class=${PROXY_PER_CLASS} epochs=${PROXY_EPOCHS} (early-stop off)"
echo "   trials     : ${ONLY}"
echo "   region=${REGION} bucket=${BUCKET}  (jobs queue if T4 quota < #trials)"
echo "   NOTE: domain_augment is new code — run ./build_and_push.sh FIRST."
echo

N=0
for id in ${ONLY}; do
  strength="${STRENGTH[$id]}"
  suffix="v4-${id}-${LABEL[$id]//[^a-zA-Z0-9]/}"
  echo ">> submitting ${id} (${LABEL[$id]}): --aug-strength=${strength}"
  JOB_SUFFIX="${suffix}" "${HERE}/submit_job.sh" "${BASE[@]}" --aug-strength="${strength}"
  N=$((N+1))
  echo
done

echo ">> submitted ${N} proxy trials."
echo "   track:   gcloud ai custom-jobs list --region=${REGION} --format='table(displayName,state)'"
echo "   compare: read each job's eval_macro_f1 + per-class US/CA F1 (higher = better):"
echo "     for j in \$(gcloud ai custom-jobs list --region=${REGION} --filter='displayName:accent-classifier-*-v4-d*' --format='value(displayName)'); do"
echo "       echo -n \"\$j  \"; gcloud storage cat gs://${BUCKET}/outputs/classifier/\$j/model/final_metrics.json 2>/dev/null | python -c 'import sys,json;d=json.load(sys.stdin);print(\"mf1\",round(d.get(\"eval_macro_f1\",0),3),\"US\",round(d.get(\"eval_f1_US\",0),3),\"CA\",round(d.get(\"eval_f1_CA\",0),3))'; done"
echo
echo ">> ACCEPTANCE (do NOT skip): full-train the winning strength, then eval on VoxForge."
echo "     JOB_SUFFIX=v4-domainaug ./submit_job.sh --auto-register \\"
echo "       --backbone=${WAVLM} --unfreeze-top=12 --lr=3e-5 --class-weight=balanced \\"
echo "       --warmup-ratio=0.15 --augment --aug-strength=<winner> \\"
echo "       --per-class=5000 --epochs=10 --early-stopping-patience=3"
echo "     # then eval on VoxForge (CPU job; env-var interface):"
echo "     #   MODEL=/gcs/${BUCKET}/outputs/classifier/<JOB>/model \\"
echo "     #   CURATED=/gcs/qi-ucsd-speech-usc1/test_voxforge \\"
echo "     #   JOB_SUFFIX=voxforge-v4-domainaug ./submit_eval_job.sh"
echo "     # ACCEPT ONLY IF VoxForge acc/macro-F1 >= v3 (0.713 / 0.698) AND CA F1 recovers."
