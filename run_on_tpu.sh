set -eux

HOST_ID="${HOSTNAME: -1}"

PYTHON_VERSION=cp38  # Supported python versions: cp36, cp37, cp38

pip install --upgrade --user https://storage.googleapis.com/jax-releases/tpu/jaxlib-0.1.55+tpu-$PYTHON_VERSION-none-manylinux2010_x86_64.whl

sudo tee -a /usr/local/lib/python3.6/dist-packages/jax_pod_setup.py > /dev/null <<EOF
import os
import requests

def get_metadata(key):
  return requests.get(
      'http://metadata.google.internal/computeMetadata'
      '/v1/instance/attributes/{}'.format(key),
      headers={
          'Metadata-Flavor': 'Google'
      }).text

worker_id = get_metadata('agent-worker-number')
accelerator_type = get_metadata('accelerator-type')
worker_network_endpoints = get_metadata('worker-network-endpoints')

os.environ['CLOUD_TPU_TASK_ID'] = worker_id
os.environ['TPU_CHIPS_PER_HOST_BOUNDS'] = '2,2,1'

accelerator_type_to_host_bounds = {
    'v2-8': '1,1,1',
    'v2-32': '2,2,1',
    'v2-128': '4,4,1',
    'v2-256': '4,8,1',
    'v2-512': '8,8,1',
    'v3-8': '1,1,1',
    'v3-32': '2,2,1',
    'v3-128': '4,4,1',
    'v3-256': '4,8,1',
    'v3-512': '8,8,1',
    'v3-1024': '8,16,1',
    'v3-2048': '16,16,1',
}

os.environ['TPU_HOST_BOUNDS'] = accelerator_type_to_host_bounds[
    accelerator_type]
os.environ['TPU_MESH_CONTROLLER_ADDRESS'] = worker_network_endpoints.split(
    ',')[0].split(':')[2] + ':8476'
os.environ['TPU_MESH_CONTROLLER_PORT'] = '8476'
EOF

pip install ipython ipdb
export PATH=/home/mweiss10/.local/bin:$PATH

sudo apt install sshfs -y

git config --global user.name "Martin Weiss"
git config --global user.email martin.clyde.weiss@gmail.com


if [ "$HOST_ID" -eq "0" ]
then
  sudo apt install emacs -y
  git clone https://github.com/google/jax.git
  cd jax
  git remote add skye https://github.com/skye/jax.git
  # optional: check out your dev branch
  cd
else
  mkdir sshfs
fi

ssh-keygen -t rsa -b 4096 -C my_tpu_pod -N '' -f ~/.ssh/id_rsa
cat .ssh/id_rsa.pub

# emacs .ssh/authorized_keys
# sshfs ...: sshfs  # Use _internal_ IP address of host 0!
# cd sshfs
# cd jax
# pip install --upgrade -e .

