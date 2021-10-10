############### BOOT GCP TPU-VM JOBS ################

# Read inputs
ROOTID=${1:-"0"}
NUMNODES=${2:-2}
EXPNAME=${3:-"experiments/t5-xs-shakespeare-test"}
TEMPLATENAME=${4:-"templates/t5-xs-shakespeare"}
NETWORK=${5:-"tpu-network"}
SUBNETWORK=${6:-"swarm-2"}
RANGE=${7:-"192.169.0.0/29"}
USER=${8:-"martin"}
echo launching $NUMNODES nodes on $NETWORK/$SUBNETWORK

# Boot control plane node
gcloud alpha compute tpus tpu-vm create "node-$ROOTID" \
  --zone us-central1-f \
  --network $NETWORK \
  --subnetwork $SUBNETWORK \
  --range $RANGE \
  --accelerator-type v2-8 \
  --version v2-alpha \
  --metadata=startup-script="#! /bin/bash
    export XRT_TPU_CONFIG='localservice;0;localhost:51011'; unset LD_PRELOAD;
  "

# Get internal-ip for the control plane node
CONTROLIP=$(gcloud alpha compute tpus describe "node-$ROOTID" --format='get(ipAddress)')
echo $CONTROLIP

# Boot all the other nodes (if booting more than one
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
  --async
done

# ssh into each node and setup polytax
for i in $(seq 0 $(($NUMNODES-1))); do
  NODENAME="node-$(($i+$ROOTID))"
  echo running startup script on $NODENAME

  gcloud alpha compute tpus tpu-vm ssh $USER@$NODENAME --zone us-central1-f --command "cd ~/; git clone https://github.com/mweiss17/polytax.git; chmod 755 polytax/src/polytax/scripts/*.sh; ./polytax/src/polytax/scripts/launch_tpu.sh $i localhost $EXPNAME $TEMPLATENAME"

done


# Poll and check for pre-empted. If pre-empted, then
#curl -sfH 'Metadata-Flavor: Google' 'http://169.254.169.254/computeMetadata/v1/instance/preempted'


# Connect to Cloud TPU
# gcloud alpha compute tpus tpu-vm ssh node-0 --zone us-central1-f


# Delete the Cloud TPU
#gcloud alpha compute tpus tpu-vm delete tpu-name  --zone us-east1-f

# create a remote coordinator:
#gcloud compute --project=must-318416 instances create coord1 \
#  --zone=us-central1-f  \
#  --network tpu-network \
#  --subnet swarm-2 \
#  --machine-type=n1-standard-1  \
#  --image-family=torch-xla \
#  --image-project=ml-images  \
#  --boot-disk-size=200GB \
#  --scopes=https://www.googleapis.com/auth/cloud-platform

# python3 -m torch_xla.distributed.xla_dist --tpu=node-0:node-1 --restart --env LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4 -- python3 /home/martin/pytorch/xla/test/test_train_mp_imagenet.py --fake_data --model=resnet50 --num_epochs=1
# sudo vi /usr/local/lib/python3.8/dist-packages/torch_xla/distributed/xla_multiprocessing.py

# need to change c_localservice to localservice

# 214: --worker {}.format("0")
# 227 --worker {}.format("0")

# insert on line 261 cluster.py
#    if ":" in tpu:
#        tpu = [t for t in tpu.split(":")]


#    if not self._is_retry() and not self.tpuvm_mode:
#      return [
#          'gcloud',
#          '-q',
#          'compute',
#          'scp',
#         '--internal-ip',
#          '--zone={}'.format(client_worker.get_zone()),
#          local_path,
#          '{}:{}'.format(client_worker.get_hostname(), remote_path),
#          ]

#
#   def _build_ssh_cmd(self, remote_cmd, client_worker):
#    services = self._cluster.get_service_workers()
#    service_map = {s.get_internal_ip(): s._tpu for s in services}
#    tpu_name = service_map[client_worker.get_internal_ip()]


#
#'{}'.format(tpu_name)
#
## 238:
#-        '{}@{}'.format(os.getlogin(), client_worker.get_hostname()),
#+        '{}@{}'.format(os.getlogin(), client_worker.get_internal_ip()),

#270: +    print(f"_build_and_run_ssh: {''.join(cmd)} on {client_worker}")

# worker_name = 'localservice' if self.tpuvm_mode else 'c_tpu_worker'

# sudo vi /usr/local/lib/python3.8/dist-packages/torch_xla/distributed/xla_dist.py
# insert on line of xla_dist.py
#
#    services = self._cluster.get_service_workers()
#    print(f"services: {services}, client_worker: {client_worker.get_internal_ip()}")
#    service_map = {s.get_internal_ip(): s._tpu for s in services}
#    tpu_name = service_map[client_worker.get_internal_ip()]
