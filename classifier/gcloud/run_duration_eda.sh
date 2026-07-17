#!/usr/bin/env bash
# Spins up a short-lived VM in $REGION next to the curated bucket, measures the
# real per-clip audio duration of every curated file with ffprobe, builds a
# histogram report, copies the small result files back locally, then deletes
# the VM. Nothing bulk-downloads to this machine (CLAUDE.md §2/§4).
set -euo pipefail
cd "$(dirname "$0")"
source env.sh

ZONE="${REGION}-a"
TS=$(date +%Y%m%d-%H%M%S)
OUT_PREFIX="reports/duration_eda/${TS}"
INSTANCE="duration-eda-${TS}"
CLASSES="US,UK,CA,AU,IN,CN"
LOCAL_OUT="../docs/assets/duration_eda_${TS}"

echo "Creating ${INSTANCE} in ${ZONE} (bucket=${BUCKET}) ..."
gcloud compute instances create "$INSTANCE" \
  --project="$PROJECT_ID" --zone="$ZONE" \
  --machine-type=e2-standard-4 \
  --image-family=debian-12 --image-project=debian-cloud \
  --boot-disk-size=30GB \
  --scopes=cloud-platform \
  --metadata-from-file startup-script=duration_eda_startup.sh,duration-eda-script=duration_eda_vm.py \
  --metadata "^;^bucket=${BUCKET};out-prefix=${OUT_PREFIX};classes=${CLASSES}"

echo "Waiting for gs://${BUCKET}/${OUT_PREFIX}/_DONE ..."
for i in $(seq 1 90); do
  if gcloud storage ls "gs://${BUCKET}/${OUT_PREFIX}/_DONE" >/dev/null 2>&1; then
    echo "Measurement done."
    break
  fi
  sleep 20
done

if ! gcloud storage ls "gs://${BUCKET}/${OUT_PREFIX}/_DONE" >/dev/null 2>&1; then
  echo "Timed out waiting for results; check /var/log/duration_eda.log on ${INSTANCE} before deleting it." >&2
  exit 1
fi

mkdir -p "$LOCAL_OUT"
gcloud storage cp "gs://${BUCKET}/${OUT_PREFIX}/*" "$LOCAL_OUT/"

echo "Deleting ${INSTANCE} ..."
gcloud compute instances delete "$INSTANCE" --zone="$ZONE" --quiet

echo "Results: $LOCAL_OUT (also kept at gs://${BUCKET}/${OUT_PREFIX}/)"
