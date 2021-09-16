import time
import torch.distributed as dist
import torch_xla.distributed.xla_multiprocessing as xmp

from argparser import get_args
from network import get_internal_ip


args = get_args()
print(args)
if not args.addr:
    print("Retrieving internal ip...")
    args.addr = get_internal_ip()
    print(args.addr)
print(f"tcp://{args.addr}:{args.port}")
# :dist.init_process_group(backend="gloo", init_method=f"tcp://{args.addr}:{args.port}", rank=args.rank, world_size=args.size)
# train(args, start)
# init_train(args)

# def launch_local(args, start):
#     processes = []
#     mp.set_start_method("spawn")
#     for rank in range(args.size):
#         p = mp.Process(target=init_process, args=(rank, args.size, run, args.addr, args.port))
#         p.start()
#         processes.append(p)
#
#     for p in processes:
#         p.join()
#
# curl –O https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
# gsutil -u must-318416 -m cp 'gs://allennlp-tensorflow-datasets/c4/realnewslike/3.0.1/*' 'gs://c4-datasets/c4/realnewslike/3.0.1/'

# pip install apache-beam[gcp]

