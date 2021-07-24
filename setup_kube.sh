# Install kubectl
gcloud components install kubectl

# Set project name and region
gcloud config set project MUST
gcloud config set compute/zone us-central1-f

# Setup GKE cluster
gcloud container clusters create tpu-cluster-1 \
  --cluster-version=1.16 \
  --scopes=cloud-platform \
  --enable-ip-alias \
  --enable-tpu


kubectl create -f switch-job.yaml