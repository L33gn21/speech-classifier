#!/usr/bin/env bash
# Startup script for the temporary duration-EDA VM (see run_duration_eda.sh).
# Fetches duration_eda_vm.py + params from instance metadata, runs it, then
# shuts the VM down. run_duration_eda.sh deletes the instance afterwards.
set -euo pipefail
exec > /var/log/duration_eda.log 2>&1

meta() { curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }

apt-get update -y
apt-get install -y ffmpeg python3-pip
pip3 install --break-system-packages --quiet google-cloud-storage pandas matplotlib

meta duration-eda-script > /tmp/duration_eda_vm.py

BUCKET=$(meta bucket)
OUT_PREFIX=$(meta out-prefix)
CLASSES=$(meta classes)

python3 /tmp/duration_eda_vm.py "$BUCKET" "$OUT_PREFIX" "$CLASSES"

shutdown -h now
