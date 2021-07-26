import os
import time
import argparse
import torch.multiprocessing as mp
import torch
import torch.distributed as dist


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true", help="Whether we should run a dev version locally")
    parser.add_argument("--rank", type=int, help="rank of this process")
    parser.add_argument("--size", type=int, help="number of processes", default=2)
    parser.add_argument("--addr", type=str, help="ip address", default="127.0.0.1")
    parser.add_argument("--port", type=str, help="ip port number", default="2345")

    return parser.parse_args()


def run(rank, size):
    """ Distributed function to be implemented later. """
    print(f"rank: {rank}, size: {size}")
    tensor = torch.randn((10000, 10000))
    print(tensor.element_size())

    start = time.time()

    if rank == 0:
        # Send the tensor to process 1
        dist.send(tensor=tensor, dst=1)
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)
    print(f"elapsed: {time.time() - start}")

    print('Rank ', rank, ' has data ', tensor[0])


def init_process(rank, size, fn, addr, port, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = addr
    os.environ['MASTER_PORT'] = port
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


def launch_local(args):
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


def main(args):
    if args.dev:
        launch_local(args)
    else:
        launch_cluster(args)

if __name__ == "__main__":
    print("Beginning main.py")
    args = get_args()
    main(args)
