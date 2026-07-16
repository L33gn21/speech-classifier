#!/usr/bin/env bash
# Copy to env.sh and fill in. env.sh is gitignored (holds your project values).
#   cp env.example.sh env.sh && edit env.sh

# --- GCP project / location -------------------------------------------------
export PROJECT_ID="your-gcp-project"
export REGION="us-west2"

# --- Cloud Storage ----------------------------------------------------------
# Bucket name only (no gs://). Holds data, manifests, and model outputs.
export BUCKET="your-bucket"

# --- Artifact Registry (Docker image) --------------------------------------
export REPO="speech-classifier"          # Artifact Registry repo name
export IMAGE="accent-classifier"
export IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}:latest"

# --- Vertex AI worker machine ----------------------------------------------
export MACHINE_TYPE="n1-standard-8"
export ACCELERATOR_TYPE="NVIDIA_TESLA_T4"   # or NVIDIA_L4 (needs g2-standard-* machine)
export ACCELERATOR_COUNT="1"

# --- Vertex AI TensorBoard ---------------------------------------------------
# Create once with: gcloud ai tensorboards create --display-name=speech-classifier --region=${REGION}
# then paste the returned resource name below. Leave empty to submit jobs
# without TensorBoard streaming.
export TENSORBOARD_ID=""
export SERVICE_ACCOUNT=""   # e.g. <PROJECT_NUMBER>-compute@developer.gserviceaccount.com
