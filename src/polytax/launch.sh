
NETWORK=${1:-"tpu-network"}
SUBNETWORK=${2:-"swarm-1"}
NUMNODES=${3:-2}
STARTNUM=${4:-"0"}
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
  --metadata=startup-script='#! /bin/bash
    apt update
    sudo su martin
    cd /home/martin

    git clone https://github.com/mweiss17/polytax.git
    cd polytax

    export RANK='"$i"'
    python3 -m pip install --upgrade build
    echo hello >> logs.txt
    pwd >> logs.txt
    ls -la >> logs.txt
    echo $RANK >> logs.txt

    python main.py'
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
