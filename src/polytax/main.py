import os
import time
import argparse
import torch.multiprocessing as mp
import torch
import torch.distributed as dist


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true", help="Whether we should run a dev version locally")
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


def launch_local():
    size = 2
    addr = '127.0.0.1'
    port = '2345'
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run, addr, port))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

def launch_cluster():
    addr = '192.168.0.2'
    port = '2345'
    init_process(rank=1, size=2, fn=run, addr=addr, port=port)


def main(args):
    if args.dev:
        launch_local()
    else:
        launch_cluster()

if __name__ == "__main__":
    args = get_args()
    main(args)
