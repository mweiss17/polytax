ROOTID=${1:-"0"}
NUMNODES=${2:-2}
NETWORK=${3:-"tpu-network"}
SUBNETWORK=${4:-"swarm-2"}
RANGE="192.169.0.0/29"
echo launching $NUMNODES nodes on $NETWORK/$SUBNETWORK
ENDNUM=$((ROOTID+$NUMNODES-2))

# Boot node-0 (which is also the control plane)
gcloud alpha compute tpus tpu-vm create "node-$ROOTID" \
  --zone us-central1-f \
  --network $NETWORK \
  --subnetwork $SUBNETWORK \
  --range $RANGE \
  --accelerator-type v2-8 \
  --version v2-alpha \
  --metadata=startup-script="#! /bin/bash
    export XRT_TPU_CONFIG='localservice;0;localhost:51011'; unset LD_PRELOAD; cd /home/root; git clone https://github.com/mweiss17/polytax.git; cd polytax; python3 -m pip install --upgrade build; cd src/polytax; python3 main.py --rank=$ROOTID --addr=$CONTROLIP --port=2345 >> /home/root/polytax/logs.txt
  "

# Get internal-ip for the controlling node
CONTROLIP=$(gcloud alpha compute tpus describe node-0 --format='get(ipAddress)')
echo $CONTROLIP
for i in $(seq 1 $(($NUMNODES-1))); do
  NODENAME="node-$(($i+$ROOTID))"
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
    export XRT_TPU_CONFIG='localservice;0;localhost:51011'; unset LD_PRELOAD; cd /home/root/; git clone https://github.com/mweiss17/polytax.git; cd polytax; python3 -m pip install --upgrade build; cd src/polytax; python3 main.py --rank=$i --addr=$CONTROLIP --port=2345 >> /home/root/polytax/logs.txt
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
