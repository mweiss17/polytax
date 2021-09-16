import os
import sys
import time
import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.test.test_utils as test_utils
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.core.xla_env_vars as xenv

from datasets import get_datasets
from models import MNISTModel
from argparser import get_args
from network import get_internal_ip
from setup_env import setup_env

def _train_update(device, x, loss, tracker, writer):
    test_utils.print_training_update(device, x, loss.item(), tracker.rate(), tracker.global_rate(), summary_writer=writer)


def train(args):
    # Set seed
    torch.manual_seed(1)
    
    train_dataset, test_dataset = get_datasets(args.datadir, args.dataset_name)
    
    train_sampler = None
    if xm.xrt_world_size() > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        drop_last=args.drop_last,
        shuffle=False if train_sampler else True,
        num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        drop_last=args.drop_last,
        shuffle=False,
        num_workers=args.num_workers)

    # Scale learning rate to num cores
    lr = args.lr * xm.xrt_world_size()

    device = xm.xla_device()
    model = MNISTModel().to(device)
    writer = None
    if xm.is_master_ordinal():
        writer = test_utils.get_summary_writer(args.logdir)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum)
    loss_fn = nn.NLLLoss()

    def train_loop_fn(loader):
        tracker = xm.RateTracker()
        model.train()
        for step, (data, target) in enumerate(loader):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            xm.optimizer_step(optimizer)
            tracker.add(args.batch_size)
            if step % args.log_steps == 0:
                xm.add_step_closure(
                    _train_update,
                    args=(device, step, loss, tracker, writer))

    def test_loop_fn(loader):
        total_samples = 0
        correct = 0
        model.eval()
        for data, target in loader:
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum()
            total_samples += data.size()[0]

        accuracy = 100.0 * correct.item() / total_samples
        accuracy = xm.mesh_reduce('test_accuracy', accuracy, np.mean)
        return accuracy

    train_device_loader = pl.MpDeviceLoader(train_loader, device)
    test_device_loader = pl.MpDeviceLoader(test_loader, device)
    accuracy, max_accuracy = 0.0, 0.0
    for epoch in range(1, args.num_epochs + 1):
        xm.master_print('Epoch {} train begin {}'.format(epoch, test_utils.now()))
        train_loop_fn(train_device_loader)
        xm.master_print('Epoch {} train end {}'.format(epoch, test_utils.now()))

        accuracy = test_loop_fn(test_device_loader)
        xm.master_print('Epoch {} test end {}, Accuracy={:.2f}'.format(
            epoch, test_utils.now(), accuracy))
        max_accuracy = max(accuracy, max_accuracy)
        test_utils.write_to_summary(
            writer,
            epoch,
            dict_to_write={'Accuracy/test': accuracy},
            write_xla_metrics=True)
        if args.metrics_debug:
            xm.master_print(met.metrics_report())

    test_utils.close_summary_writer(writer)
    xm.master_print('Max Accuracy: {:.2f}%'.format(max_accuracy))
    return max_accuracy

def _mp_fn(index, args):
    torch.set_default_tensor_type('torch.FloatTensor')
    print(f"index: {index}")
    print(f"xm index: {xm.get_ordinal()}")
    print(f"xm local index: {xm.get_local_ordinal()}")
    
    # dist.init_process_group(backend="gloo", init_method=f"tcp://{args.addr}:{args.port}", rank=index, world_size=args.size)
    #print(f"index: {index}, xmtorch dist rank: {dist.get_rank()}")
    accuracy = train(args)
    print(f"accuracy: {accuracy}")
    sys.exit(21)


if __name__ == '__main__':
    args = get_args()
    print(args)
    if not args.addr:
        print("Retrieving internal ip...")
        args.addr = get_internal_ip()
        print(args.addr)
    print(f"tcp://{args.addr}:{args.port}")
    setup_env(args)
    print(os.environ[xenv.HOST_ORDINAL])    
    xmp.spawn(_mp_fn, args=(args,), nprocs=args.ncores, start_method='fork')
