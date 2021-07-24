import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run(rank, size):
    """ Distributed function to be implemented later. """
    print(f"rank: {rank}, size: {size}")
    tensor=torch.randn((100, 100))
    # print(tensor.element_size() * tensor.nelement())
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


def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] =  '192.168.0.2' #'127.0.0.1'
    os.environ['MASTER_PORT'] = '2345'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

if __name__ == "__main__":
    init_process(rank=1, size=2, fn=run)
#
# if __name__ == "__main__":
#     size = 2
#     processes = []
#     mp.set_start_method("spawn")
#     for rank in range(size):
#         p = mp.Process(target=init_process, args=(rank, size, run))
#         p.start()
#         processes.append(p)
#
#     for p in processes:
#         p.join()
