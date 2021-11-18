from torchvision import datasets, transforms

import os
import textfile_utils

TASK_ROOT_DIR = textfile_utils.TASK_ROOT_DIR

DATA_DIR = os.path.join(TASK_ROOT_DIR, 'data')

_MNIST_TRAIN_TRANSFORMS = _MNIST_TEST_TRANSFORMS = [
    transforms.ToTensor(),
    transforms.ToPILImage(),
    transforms.Pad(2),
    transforms.ToTensor(),
]

_CIFAR10_TRAIN_TRANSFORMS = _CIFAR10_TEST_TRANSFORMS = [
    transforms.ToTensor(),
    #transforms.Normalize([0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010]),
]

TRAIN_DATASETS = {
    'MNIST': datasets.MNIST(
        DATA_DIR + '/mnist', train=True, download=True,
        transform=transforms.Compose(_MNIST_TRAIN_TRANSFORMS)
    ),
    'CIFAR10': datasets.CIFAR10(
        DATA_DIR + '/cifar10', train=True, download=True,
        transform=transforms.Compose(_CIFAR10_TRAIN_TRANSFORMS)
    ),
}


TEST_DATASETS = {
    'MNIST': datasets.MNIST(
        DATA_DIR + '/mnist', train=False,
        transform=transforms.Compose(_MNIST_TEST_TRANSFORMS)
    ),
    'CIFAR10': datasets.CIFAR10(
        DATA_DIR + '/cifar10', train=False,
        transform=transforms.Compose(_CIFAR10_TRAIN_TRANSFORMS)
    ),
}


DATASET_CONFIGS = {
    'MNIST': {'size': 32, 'channels': 1, 'classes': 10},
    #'MNIST': {'size': 28, 'channels': 1, 'classes': 10},
    'CIFAR10': {'size': 32, 'channels': 3, 'classes': 10},
}
