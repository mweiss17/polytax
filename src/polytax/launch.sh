
NETWORK=${1:-"tpu-network"}
SUBNETWORK=${2:-"swarm-1"}
NUMNODES=${3:-2}
echo launching $NUMNODES nodes on $NETWORK/$SUBNETWORK
break
for i in $(seq 1 $NUMNODES); do
  NODENAME="node-$(($i-1))"
  echo spooling up $NODENAME

  # Spool up a TPU node
  gcloud alpha compute tpus tpu-vm create $NODENAME \
  --zone us-central1-f \
  --network $NETWORK \
  --subnetwork $SUBNETWORK \
  --accelerator-type v2-8 \
  --version v2-alpha \
  --async \
  --metadata-from-file=startup-script=startup.sh \
  --metadata=RANK=$i
done

#break
## Connect to Cloud TPU
## gcloud alpha compute tpus tpu-vm ssh node-0 --zone us-central1-f
#
## SSH into the Cloud TPU
#pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
#
## Delete the Cloud TPU
#gcloud alpha compute tpus tpu-vm delete tpu-name  --zone us-east1-f
#
#out = pmap(lambda x: x ** 2)(np.arange(8))
#print(out)
