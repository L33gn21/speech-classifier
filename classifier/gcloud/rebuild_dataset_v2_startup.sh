#!/usr/bin/env bash
# Startup script for the temporary dataset-rebuild VM (see rebuild_dataset_v2.sh).
# Fetches rebuild_dataset_v2_vm.py + params from instance metadata, runs the
# full resample/split/rebalance pipeline straight into the bucket, then shuts
# the VM down. rebuild_dataset_v2.sh deletes the instance afterwards.
set -euo pipefail
exec > /var/log/rebuild_dataset_v2.log 2>&1

meta() { curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }

apt-get update -y
apt-get install -y python3-pip ffmpeg libsndfile1

pip3 install --break-system-packages --quiet \
  google-cloud-storage "soundfile>=0.12" "librosa>=0.10" "numpy>=1.26"

meta rebuild-script > /tmp/rebuild_dataset_v2_vm.py

BUCKET=$(meta bucket)
LOG_PREFIX=$(meta log-prefix)
SOURCE_PREFIX=$(meta source-prefix || echo "curated")

python3 /tmp/rebuild_dataset_v2_vm.py "$BUCKET" "$LOG_PREFIX" "$SOURCE_PREFIX"

shutdown -h now
