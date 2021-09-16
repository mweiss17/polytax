# https://cloud.google.com/vpc/docs/using-vpc
gcloud auth login martin
gcloud compute networks create tpu-network \
    --subnet-mode=auto \
    --bgp-routing-mode=regional \
    --mtu=1460

gcloud compute firewall-rules create tpu-network-firewall --network tpu-network --allow tcp,udp,icmp #--source-ranges <IP_RANGE>

gcloud compute networks subnets create swarm-2 --network=tpu-network     --range=192.169.0.0/16     --region=us-central1

gcloud compute networks subnets delete swarm-1 --region=us-central1-f
