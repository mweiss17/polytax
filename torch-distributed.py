import torch.distributed as dist
dist.is_available()

backend = "GLOO"
rank = 0
dist.init_process_group(backend, init_method='tcp://192.168.0.2:2345', rank=rank, world_size=2)
