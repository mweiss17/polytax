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

from datasets import get_datasets
from models import MNISTModel

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
    accuracy = train(args)
    print(f"accuracy: {accuracy}")
    sys.exit(21)

  # if flags.tidy and os.path.isdir(flags.datadir):
  #   shutil.rmtree(flags.datadir)

# def _mp_fn(index):
#     device = xm.xla_device()
#     mp_device_loader = pl.MpDeviceLoader(train_loader, device)
#
#     model = MNISTModel().train().to(device)
#     loss_fn = nn.NLLLoss()
#     optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum)
#
#     for data, target in mp_device_loader:
#         optimizer.zero_grad()
#         output = model(data)
#         loss = loss_fn(output, target)
#         loss.backward()
#             xm.optimizer_step(optimizer)


def run_comm(rank, size, start):
    """ Working example function to try out torch distributed. """
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

