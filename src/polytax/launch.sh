STARTNUM=${1:-"1"}
NUMNODES=${2:-2}
NETWORK=${3:-"tpu-network"}
SUBNETWORK=${4:-"swarm-2"}
RANGE="192.169.0.0/29"
echo launching $NUMNODES nodes on $NETWORK/$SUBNETWORK
ENDNUM=$(($STARTNUM+$NUMNODES-2))

# Boot node-0 (which is also the control plane)
gcloud alpha compute tpus tpu-vm create node-0 \
  --zone us-central1-f \
  --network $NETWORK \
  --subnetwork $SUBNETWORK \
  --range $RANGE \
  --accelerator-type v2-8 \
  --version v2-alpha \
  --metadata=startup-script="#! /bin/bash
    sudo useradd -m martin;
    sudo -u martin bash -c 'cd ~/; git clone https://github.com/mweiss17/polytax.git; cd polytax; python3 -m pip install --upgrade build; cd src/polytax; python3 main.py --rank=$i --port=2345 >> /home/martin/polytax/logs.txt'"

# Get internal-ip for the controlling node
CONTROLIP=$(gcloud compute instances describe node-0 --format='get(networkInterfaces[0].networkIP)')
echo $CONTROLIP
for i in $(seq $STARTNUM $ENDNUM); do
  NODENAME="node-$(($i))"
  echo creating $NODENAME

  # Spool up a TPU node
  gcloud alpha compute tpus tpu-vm create $NODENAME \
  --zone us-central1-f \
  --network $NETWORK \
  --subnetwork $SUBNETWORK \
  --range $RANGE \
  --accelerator-type v2-8 \
  --version v2-alpha \
  --metadata=startup-script="#! /bin/bash
    sudo useradd -m martin
    sudo -u martin bash -c 'cd ~/; git clone https://github.com/mweiss17/polytax.git; cd polytax; python3 -m pip install --upgrade build; cd src/polytax; python3 main.py --rank=$i --addr=$CONTROLIP --port=2345 >> /home/martin/polytax/logs.txt'
    " \
  --async

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
