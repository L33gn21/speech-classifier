#!/usr/bin/env bash
# Spins up a short-lived VM in $REGION next to the bucket, streams the ASVspoof
# 2019 LA dataset from HuggingFace straight into the bucket (curated_spoof
# schema, protocol splits preserved), then deletes the VM. Nothing bulk-
# downloads to this machine (CLAUDE.md 2/4). HF->VM is GCP ingress (free);
# VM->GCS is same-region internal (free).
set -euo pipefail
cd "$(dirname "$0")"
source env.sh

ZONE="${REGION}-a"
TS=$(date +%Y%m%d-%H%M%S)
INSTANCE="ingest-asvspoof-${TS}"
DEST_ROOT="curated_spoof/asvspoof2019_la"
HF_DATASET="Bisher/ASVspoof_2019_LA"

echo "Creating ${INSTANCE} in ${ZONE} (bucket=${BUCKET}, dest=${DEST_ROOT}) ..."
gcloud compute instances create "$INSTANCE" \
  --project="$PROJECT_ID" --zone="$ZONE" \
  --machine-type=e2-standard-4 \
  --image-family=debian-12 --image-project=debian-cloud \
  --boot-disk-size=80GB \
  --scopes=cloud-platform \
  --metadata-from-file startup-script=ingest_asvspoof_startup.sh,ingest-script=ingest_asvspoof_vm.py \
  --metadata "^;^bucket=${BUCKET};dest-root=${DEST_ROOT};hf-dataset=${HF_DATASET}"

echo "Waiting for gs://${BUCKET}/${DEST_ROOT}/_DONE (polling up to ~60 min) ..."
DONE=0
for i in $(seq 1 180); do
  if gcloud storage ls "gs://${BUCKET}/${DEST_ROOT}/_DONE" >/dev/null 2>&1; then
    echo "Ingest done."
    DONE=1
    break
  fi
  sleep 20
done

if [ "$DONE" -ne 1 ]; then
  echo "Timed out waiting for _DONE; do NOT auto-delete — inspect the VM log first:" >&2
  echo "  gcloud compute ssh ${INSTANCE} --zone=${ZONE} --command='sudo tail -n 100 /var/log/ingest_asvspoof.log'" >&2
  exit 1
fi

echo "----- ingest_report.json -----"
gcloud storage cat "gs://${BUCKET}/${DEST_ROOT}/ingest_report.json" || true
echo "------------------------------"

echo "Deleting ${INSTANCE} ..."
gcloud compute instances delete "$INSTANCE" --zone="$ZONE" --quiet

echo "Done. Data at gs://${BUCKET}/${DEST_ROOT}/{train,dev,eval}/"
