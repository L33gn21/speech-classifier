#!/usr/bin/env bash
# One-off: submit CPU-only evaluate.py jobs in parallel for the 10 mtv2 sweep
# models (2026-07-23-mt-v2-a100-sweep.md), scoring each on ITS OWN test.csv
# (already sitting in <model>/manifests/, written by train.py at train time).
# Uses n1-standard-8 (no accelerator) -- cheap CPU custom job, not the
# a2-highgpu-1g in env.sh (that's GPU-only, would error with no accelerator).
#
# Region must match each job's own output bucket region (Vertex constraint):
#   qi-ucsd-speech-usc1-a100 -> us-central1
#   qi-ucsd-speech-usw3-a100 -> us-west3
set -euo pipefail
PROJECT_ID="qi-ucsd-project"
IMAGE_URI="us-west2-docker.pkg.dev/${PROJECT_ID}/speech-classifier/accent-classifier:latest"
MACHINE_TYPE="n1-standard-8"

gcloud config set project "${PROJECT_ID}" >/dev/null

submit_one() {
  local region="$1" bucket="$2" job="$3"
  local model="/gcs/${bucket}/outputs/classifier/${job}/model"
  local manifest="${model}/manifests"
  local job_name="eval-$(date +%Y%m%d-%H%M%S)-${job}"
  local out_base="gs://${bucket}/outputs/classifier/${job_name}"
  local cfg
  cfg="$(mktemp)"
  cat > "${cfg}" <<YAML
baseOutputDirectory:
  outputUriPrefix: ${out_base}
workerPoolSpecs:
  - machineSpec:
      machineType: ${MACHINE_TYPE}
    replicaCount: 1
    containerSpec:
      imageUri: ${IMAGE_URI}
      command:
        - "python"
        - "evaluate.py"
      args:
        - "--model-dir=${model}"
        - "--manifest-dir=${manifest}"
YAML
  echo ">> submitting ${job_name} (region=${region})"
  # NOTE: do NOT rm the config file here -- the gcloud call above is
  # backgrounded (&), so an immediate rm races it and deletes the file
  # before it's read (bit us the first run: "Unable to read file tmp.xxx").
  # Clean up temp configs once, after `wait`, at the bottom of the script.
  gcloud ai custom-jobs create \
    --region="${region}" \
    --display-name="${job_name}" \
    --config="${cfg}" \
    --format="value(name)" &
}

# us-central1 (qi-ucsd-speech-usc1-a100) -- 4 jobs
submit_one us-central1 qi-ucsd-speech-usc1-a100 accent-classifier-20260723-004350-mtv2-base-a100
submit_one us-central1 qi-ucsd-speech-usc1-a100 accent-classifier-20260723-004413-mtv2-lam05-a100
submit_one us-central1 qi-ucsd-speech-usc1-a100 accent-classifier-20260723-043603-mtv2-augoff-lrlow-a100
submit_one us-central1 qi-ucsd-speech-usc1-a100 accent-classifier-20260723-043625-mtv2-augoff-wd05-a100

# us-west3 (qi-ucsd-speech-usw3-a100) -- 6 jobs
submit_one us-west3 qi-ucsd-speech-usw3-a100 accent-classifier-20260723-010008-mtv2-lam20-a100
submit_one us-west3 qi-ucsd-speech-usw3-a100 accent-classifier-20260723-010027-mtv2-augoff-a100
submit_one us-west3 qi-ucsd-speech-usw3-a100 accent-classifier-20260723-043642-mtv2-augoff-lrwd-a100
submit_one us-west3 qi-ucsd-speech-usw3-a100 accent-classifier-20260723-043657-mtv2-augoff-lam07-a100
submit_one us-west3 qi-ucsd-speech-usw3-a100 accent-classifier-20260723-062421-mtv2-augoff-uf6-a100
submit_one us-west3 qi-ucsd-speech-usw3-a100 accent-classifier-20260723-062436-mtv2-augoff-drop02-a100

wait
rm -f /tmp/tmp.*  # best-effort cleanup of the per-job config temp files
echo "all 10 eval jobs submitted."
