import os
import time
import argparse
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch_xla.core.xla_model as xm

from network import get_internal_ip

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true", help="Whether we should run a dev version locally")
    parser.add_argument("--rank", type=int, help="rank of this process")
    parser.add_argument("--size", type=int, help="number of processes", default=2)
    parser.add_argument("--addr", type=str, help="ip address", default="127.0.0.1")
    parser.add_argument("--port", type=str, help="ip port number", default="2345")

    return parser.parse_args()


def run(rank, size, start):
    """ Distributed function to be implemented later. """
    print(f"rank: {rank}, size: {size}")
    tensor = torch.randn((10000, 10000))
    print(tensor.element_size())

    if rank == 0:
        # Send the tensor to process 1
        dist.send(tensor=tensor, dst=1)
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)
    print(f" {time.time() - start} elapsed after one communication")

    print('Rank ', rank, ' has data ', tensor[0])


def init_process(rank, size, fn, addr, port, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = addr
    os.environ['MASTER_PORT'] = port
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


def launch_local(args, start):
    processes = []
    mp.set_start_method("spawn")
    for rank in range(args.size):
        p = mp.Process(target=init_process, args=(rank, args.size, run, args.addr, args.port))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

def launch_cluster(args):
    init_process(rank=args.rank, size=args.size, fn=run, addr=args.addr, port=args.port)


if __name__ == "__main__":
    start = time.time()
    args = get_args()
    print(args)
    dev = xm.xla_device()
    print(f"dev: {dev}")
    if not args.dev and not args.addr:
        print("Retrieving internal ip...")
        args.addr = get_internal_ip()
        print(args.addr)
    print(f"Entering main after {time.time()-start}")
    if args.dev:
        launch_local(args, start)
    else:
        launch_cluster(args, start)
