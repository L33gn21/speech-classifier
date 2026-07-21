#!/usr/bin/env bash
# Startup script for the temporary ASVspoof-ingest VM (see ingest_asvspoof.sh).
# Fetches ingest_asvspoof_vm.py + params from instance metadata, streams the HF
# dataset straight into the bucket, then shuts the VM down. ingest_asvspoof.sh
# deletes the instance afterwards.
set -euo pipefail
exec > /var/log/ingest_asvspoof.log 2>&1

meta() { curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }

apt-get update -y
apt-get install -y python3-pip
# decode=False path needs no soundfile/librosa; datasets streams parquet + raw
# audio bytes. google-cloud-storage writes straight to the bucket.
pip3 install --break-system-packages --quiet \
  "datasets>=2.16" "huggingface_hub>=0.20" google-cloud-storage

meta ingest-script > /tmp/ingest_asvspoof_vm.py

BUCKET=$(meta bucket)
DEST_ROOT=$(meta dest-root)
HF_DATASET=$(meta hf-dataset)

python3 /tmp/ingest_asvspoof_vm.py "$BUCKET" "$DEST_ROOT" "$HF_DATASET"

shutdown -h now
