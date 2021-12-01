import os
import torch
global xla_found

try:
    import torch_xla.core.xla_model as xm
    xla_found = True
except ImportError:
    xm = None
    xla_found = False


device = xm.xla_device() if xla_found else torch.device("cpu")
LOCAL_WORLD_SIZE = int(xm.xrt_world_size()) if xla_found else 1  # Devices per host
GLOBAL_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))  # Total number of hosts
NUM_SHARDS = GLOBAL_WORLD_SIZE * LOCAL_WORLD_SIZE
LOCAL_RANK = int(xm.get_ordinal()) if xla_found else 0
GLOBAL_RANK = int(os.environ.get("RANK", 0)) * LOCAL_WORLD_SIZE + LOCAL_RANK
IS_MASTER_ORDINAL = xm.is_master_ordinal() if xla_found else True
IS_MULTI_HOST = GLOBAL_WORLD_SIZE > 1