STARTNUM=${1:-"0"}
NUMNODES=${2:-2}
NETWORK=${3:-"tpu-network"}
SUBNETWORK=${4:-"swarm-1"}
echo launching $NUMNODES nodes on $NETWORK/$SUBNETWORK
ENDNUM=$(($STARTNUM+$NUMNODES-1))
for i in $(seq $STARTNUM $ENDNUM); do
  NODENAME="node-$(($i))"
  echo creating $NODENAME

  # Spool up a TPU node
  gcloud alpha compute tpus tpu-vm create $NODENAME \
  --zone us-central1-f \
  --network $NETWORK \
  --subnetwork $SUBNETWORK \
  --accelerator-type v2-8 \
  --version v2-alpha \
  --async \
  --metadata=startup-script="#! /bin/bash
    sudo useradd -m martin
    sudo -u martin bash -c 'cd ~/ '
    sudo -u martin bash -c 'git clone https://github.com/mweiss17/polytax.git'
    sudo -u martin bash -c 'cd polytax'
    sudo -u martin bash -c 'python3 -m pip install --upgrade build'
    sudo -u martin bash -c 'cd src/polytax'
    sudo -u martin bash -c 'python3 main.py --rank=$i --addr=192.168.0.2 --port=2345 >> /home/martin/polytax/logs.txt'
    "
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
