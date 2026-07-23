#!/usr/bin/env bash
# Copy to env.sh and fill in. env.sh is gitignored (holds your project values).
#   cp env.example.sh env.sh && edit env.sh

# --- GCP project / location -------------------------------------------------
export PROJECT_ID="qi-ucsd-project"
# Home region (us-west2). We briefly ran compute in us-central1 during a us-west2
# T4 stockout; reverted to us-west2 (2026-07-17) now that its T4 quota is
# available (verified effectiveLimit=1). Data, image, and outputs all live in
# us-west2. Image is rebuilt in whichever REGION this is set to.
export REGION="us-west2"

# --- Cloud Storage ----------------------------------------------------------
# TEMP (2026-07-23, A100 run): A100 quota (2) is only granted in us-west2, but
# Vertex Custom Training doesn't support A100 in us-west2 at all (confirmed via
# job submission error). A100 IS supported in us-central1/us-west1/us-west3/
# us-west4, so we copy the multitask data (real_fake_5k, ~4.6GB) into a new
# us-central1 bucket and run there. Revert to us-west2/qi-ucsd-speech-usw2 once
# this A100 round is done.
export BUCKET="qi-ucsd-speech-usw2"

# --- Artifact Registry (Docker image) --------------------------------------
# Image stays in us-west2 registry (image location doesn't need to match job
# region, only the output bucket does) — avoids a redundant rebuild.
export REPO="speech-classifier"          # Artifact Registry repo name
export IMAGE="accent-classifier"
export IMAGE_URI="us-west2-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}:latest"

# --- Vertex AI worker machine ----------------------------------------------
# A100 (40GB) run: quota confirmed via `gcloud alpha services quota list`
# (effectiveLimit=2 in us-west2, but see BUCKET note above re: region).
export MACHINE_TYPE="n1-standard-8"
export ACCELERATOR_TYPE="NVIDIA_TESLA_A100"
export ACCELERATOR_COUNT="1"

# --- Vertex AI TensorBoard ---------------------------------------------------
# Training/eval curves (loss, accuracy, per-class F1) stream here live during
# the job. View at: https://console.cloud.google.com/vertex-ai/experiments/tensorboard-instances?project=${PROJECT_ID}
# The old us-central1 instance was DELETED in the 2026-07-16 decommission
# (see reports/2026-07-16-us-central1-decommission.md). Recreate it in us-west2
# and paste the returned resource name below. Left empty so submit_job.sh submits
# WITHOUT TensorBoard until it's recreated:
#   gcloud ai tensorboards create --display-name=speech-classifier --region=us-west2
export TENSORBOARD_ID=""
export SERVICE_ACCOUNT="296104341604-compute@developer.gserviceaccount.com"

export API_KEY="06e9a39a2f69587f6e3e332779f7ff44"