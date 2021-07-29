import os
from torchvision import datasets, transforms
import torch_xla.core.xla_model as xm

def get_datasets(datadir, name):
    print(os.path.join(datadir, str(xm.get_ordinal())))
    if name == "MNIST":
        train_dataset = datasets.MNIST(
            os.path.join(datadir, str(xm.get_ordinal())),
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,))]))
        test_dataset = datasets.MNIST(
            os.path.join(datadir, str(xm.get_ordinal())),
            train=False,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,))]))
    return train_dataset, test_dataset
