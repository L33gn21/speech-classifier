#!/usr/bin/env bash
# Rebuilds the two production data pools in place (us-west2 bucket):
#   curated/<CC>/                 -- 16kHz mono, SAA long clips split into
#                                    GLOBE-like windows, US/UK/CA capped at
#                                    5000 (speaker-stratified), AU/IN/CN kept
#                                    as-is (best effort, no forced padding)
#   curated_spoof/real_fake_5k/   -- flat real(35000)/fake(35000) pool:
#                                    real = ASVspoof bonafide 5000 + 6
#                                    countries x 5000 (AU/IN/CN topped up by
#                                    server-side duplicate copy of existing
#                                    clips); fake = ASVspoof spoof 35000
#                                    (stratified over split x system_id)
#
# Existing curated/ and curated_spoof/asvspoof2019_la/ are ARCHIVED first
# (gcloud storage cp -r, same-bucket, no egress) before curated/ is deleted
# and rebuilt. curated_spoof/asvspoof2019_la/ itself is left in place (it is
# read-only *source* data for this rebuild, not overwritten) -- only backed
# up for extra safety.
#
# Processing runs on a short-lived GCE VM in us-west2 (same region as the
# bucket -> no egress cost), never through this machine (CLAUDE.md §2/§4).
set -euo pipefail
cd "$(dirname "$0")"
source env.sh

ZONE="${REGION}-a"
TS=$(date +%Y%m%d-%H%M%S)
INSTANCE="rebuild-dataset-v2-${TS}"
LOG_PREFIX="reports/dataset_rebuild_v2/${TS}"

ARCHIVE_COUNTRY="_archive/curated_country_prev_${TS}"
ARCHIVE_SPOOF="_archive/curated_spoof_prev_${TS}"

echo "=== 1) archive existing curated/ -> gs://${BUCKET}/${ARCHIVE_COUNTRY}/ ==="
gcloud storage cp -r "gs://${BUCKET}/curated" "gs://${BUCKET}/${ARCHIVE_COUNTRY}"

echo "=== 2) archive existing curated_spoof/asvspoof2019_la/ -> gs://${BUCKET}/${ARCHIVE_SPOOF}/ ==="
gcloud storage cp -r "gs://${BUCKET}/curated_spoof/asvspoof2019_la" \
  "gs://${BUCKET}/${ARCHIVE_SPOOF}/asvspoof2019_la"

echo "=== 3) delete curated/ (archived above; about to be rebuilt fresh) ==="
gcloud storage rm -r "gs://${BUCKET}/curated"

echo "Creating ${INSTANCE} in ${ZONE} (bucket=${BUCKET}) ..."
gcloud compute instances create "$INSTANCE" \
  --project="$PROJECT_ID" --zone="$ZONE" \
  --machine-type=n1-standard-16 \
  --image-family=debian-12 --image-project=debian-cloud \
  --boot-disk-size=50GB \
  --scopes=cloud-platform \
  --metadata-from-file startup-script=rebuild_dataset_v2_startup.sh,rebuild-script=rebuild_dataset_v2_vm.py \
  --metadata "^;^bucket=${BUCKET};log-prefix=${LOG_PREFIX};source-prefix=${ARCHIVE_COUNTRY}"

echo "Waiting for gs://${BUCKET}/${LOG_PREFIX}/_DONE (polling every 10s, up to ~60 min) ..."
DONE=0
for i in $(seq 1 360); do
  if gcloud storage ls "gs://${BUCKET}/${LOG_PREFIX}/_DONE" >/dev/null 2>&1; then
    echo "Rebuild done."
    DONE=1
    break
  fi
  sleep 10
done

if [ "$DONE" -ne 1 ]; then
  echo "Timed out waiting for _DONE; do NOT auto-delete -- inspect the VM log first:" >&2
  echo "  gcloud compute ssh ${INSTANCE} --zone=${ZONE} --command='sudo tail -n 150 /var/log/rebuild_dataset_v2.log'" >&2
  exit 1
fi

echo "----- rebuild_report.json -----"
gcloud storage cat "gs://${BUCKET}/${LOG_PREFIX}/rebuild_report.json" || true
echo "--------------------------------"

echo "Deleting ${INSTANCE} ..."
gcloud compute instances delete "$INSTANCE" --zone="$ZONE" --quiet

echo "Done."
echo "  curated/                       -> gs://${BUCKET}/curated/"
echo "  curated_spoof/real_fake_5k/     -> gs://${BUCKET}/curated_spoof/real_fake_5k/"
echo "  archived (old country pool)     -> gs://${BUCKET}/${ARCHIVE_COUNTRY}/"
echo "  archived (old asvspoof copy)    -> gs://${BUCKET}/${ARCHIVE_SPOOF}/"
