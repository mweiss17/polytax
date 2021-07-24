
NETWORK="tpu-network"
SUBNETWORK="swarm-1"
NODENAME="node-1"

# Spool up a TPU node
gcloud alpha compute tpus tpu-vm create $NODENAME \
--zone us-central1-f \
--network $NETWORK \
--subnetwork $SUBNETWORK \
--accelerator-type v2-8 \
--version v2-alpha \
--async

# Connect to Cloud TPU
gcloud alpha compute tpus tpu-vm ssh node-0 --zone us-central1-f

# SSH into the Cloud TPU
pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Delete the Cloud TPU
gcloud alpha compute tpus tpu-vm delete tpu-name  --zone us-east1-f

out = pmap(lambda x: x ** 2)(np.arange(8))
print(out)
