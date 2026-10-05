#!/usr/bin/env bash
# Preflight — verify the curated training pool is present in the bucket.
#
# There is no "upload data" step: the curated pool is built VM->bucket and lives
# under gs://<BUCKET>/curated/ (see CLAUDE.md §2 and DATASET.md). train.py builds
# its speaker-disjoint train/val/test splits from these manifests at job start, so
# nothing is uploaded from local. This script just sanity-checks that the classes
# the model trains on (US/UK/IN/NG, see src/config.py LABELS) are there and prints
# per-class clip counts before you build the image / submit a job.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/env.sh"

CURATED="gs://${BUCKET}/curated"
echo ">> checking ${CURATED}  (training classes: US UK IN NG)"

TOTAL=0
MISSING=0
for CC in US UK IN NG; do
  MANIFEST="${CURATED}/${CC}/manifest.csv"
  if ! gsutil -q stat "${MANIFEST}" 2>/dev/null; then
    echo "   ${CC}: MISSING -> ${MANIFEST}"
    MISSING=$((MISSING + 1))
    continue
  fi
  # rows minus the header line = clip count
  ROWS=$(gsutil cat "${MANIFEST}" | tail -n +2 | wc -l | tr -d ' ')
  echo "   ${CC}: ${ROWS} clips"
  TOTAL=$((TOTAL + ROWS))
done

echo ">> total: ${TOTAL} clips across US/UK/IN/NG"
if [ "${MISSING}" -gt 0 ]; then
  echo ">> ${MISSING} class(es) missing — fix the curated pool before submitting."
  exit 1
fi
echo ">> ok. next:  ./build_and_push.sh   then   ./submit_job.sh"
