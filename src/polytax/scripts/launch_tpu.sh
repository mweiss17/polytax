#!/bin/bash

# Distributed training stuff -- not usable yet
rank=$1
addr=$2
expname=$3
templatename=$4
echo rank: $rank, addr: $addr, experiment_name: $expname, template_name: $templatename

cd ~/polytax/
pip3 install -e . # for some reason this fails if we python3 setup.py install --user

cd ~/polytax/src/polytax
export WANDB_API_KEY=$(curl "http://metadata.google.internal/computeMetadata/v1/project/attributes/wandb_api_key" -H "Metadata-Flavor: Google")

git config --global user.email "martin.clyde.weiss@gmail.com"
git config --global user.name "Martin Weiss"


export PATH=$PATH:/home/$USER/.local/bin
unset LD_PRELOAD
export XRT_TPU_CONFIG="localservice;0;localhost:51011"

python3 train.py experiments/$expname --inherit templates/$templatename/

# Once we get some distributed stuff working...
# python3 launch.py --rank=$rank --addr=$addr --port=2345

import os
from google.cloud import secretmanager

project_id = "must-318416"
secret_id = "TPU-Github"
parent = f"projects/must-318416"
client = secretmanager.SecretManagerServiceClient()

# List all secrets.
for secret in client.list_secrets(request={"parent": parent}):
    print("Found secret: {}".format(secret.name))
    print(secret)

name = client.secret_path(secret.name, "/versions/latest")
response = client.get_secret(request={"name": name})


response = client.create_secret(secret_id="github", parent=parent, secret=secret)


versionname = client.secret_path(project_id, secret_id) + "/versions/latest"
response = client.access_secret(request={"name": "projects/750902964106/secrets/github"})


# Import the Secret Manager client library.
from google.cloud import secretmanager

# GCP project in which to store secrets in Secret Manager.
project_id = "must-318416"

# ID of the secret to create.
secret_id = "TEMP"

# Create the Secret Manager client.
client = secretmanager.SecretManagerServiceClient()

# Build the parent name from the project.
parent = f"projects/{project_id}"

# Create the parent secret.
secret = client.create_secret(
    request={
        "parent": parent,
        "secret_id": secret_id,
        "secret": {"replication": {"automatic": {}}},
    }
)

# Add the secret version.
version = client.add_secret_version(
    request={"parent": secret.name, "payload": {"data": b"hello world!"}}
)

# Access the secret version.
response = client.access_secret_version(request={"name": version.name})

# Print the secret payload.
#
# WARNING: Do not print the secret in a production environment - this
# snippet is showing how to access the secret material.
payload = response.payload.data.decode("UTF-8")
print("Plaintext: {}".format(payload))

