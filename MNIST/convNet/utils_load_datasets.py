import os
import textfile_utils
import torch
from torchvision import datasets, transforms
from collections import OrderedDict

TASK_ROOT_DIR = textfile_utils.TASK_ROOT_DIR
MNIST_DIR = os.path.join(TASK_ROOT_DIR, 'data', 'mnist')
CIFAR10_DIR = os.path.join(TASK_ROOT_DIR, 'data', 'cifar10')

_MNIST_TRAIN_TRANSFORMS = _MNIST_TEST_TRANSFORMS = [
    transforms.ToTensor(),
    transforms.ToPILImage(),
    transforms.Pad(2),
    transforms.ToTensor(),
]

_CIFAR10_TRAIN_TRANSFORMS = _CIFAR10_TEST_TRANSFORMS = [
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    #transforms.Normalize([0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010]),
]

def load_MNIST_data(batch_size, kwargs, train_datafile=None, test_datafile = None, image_size = 32, num_classes=10, channel_num=1, conv_size=16):

    train_loader = torch.utils.data.DataLoader(datasets.MNIST(MNIST_DIR, train=True, download=True, transform=transforms.Compose(_MNIST_TRAIN_TRANSFORMS)), batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST(MNIST_DIR, train=False, transform=transforms.Compose(_MNIST_TEST_TRANSFORMS)), batch_size=batch_size, shuffle=True, **kwargs)

    dataset_info = OrderedDict()
    dataset_info['task_index'] = 1
    dataset_info['image_size'] = image_size
    dataset_info['num_classes'] = num_classes
    dataset_info['channel_num'] = channel_num
    dataset_info['conv_size'] = conv_size

    return train_loader, test_loader, dataset_info

def load_CIFAR10_data(batch_size, kwargs, train_datafile=None, test_datafile = None, image_size = 32, num_classes=10, channel_num=3, conv_size=16):

    train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(CIFAR10_DIR, train=True, download=True, transform=transforms.Compose(_CIFAR10_TRAIN_TRANSFORMS)), batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(datasets.CIFAR10(CIFAR10_DIR, train=False, transform=transforms.Compose(_CIFAR10_TEST_TRANSFORMS)), batch_size=batch_size, shuffle=True, **kwargs)

    dataset_info = OrderedDict()
    dataset_info['task_index'] = 1
    dataset_info['image_size'] = image_size
    dataset_info['num_classes'] = num_classes
    dataset_info['channel_num'] = channel_num
    dataset_info['conv_size'] = conv_size

    return train_loader, test_loader, dataset_info
