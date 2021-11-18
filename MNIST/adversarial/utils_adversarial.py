import os,sys
from adversarial import dataset_utils
from robustness.model_utils import make_and_restore_model
from robustness.datasets import DATASETS

def load_robust_model(data_path=None, dataset='MNIST', arch='resnet50', resume_path=None):
    if dataset=='MNIST':
        ds = dataset_utils.MNIST(data_path=data_path)
    elif dataset=='CIFAR10':
        ds = DATASETS['cifar'](data_path=data_path)

    model, _ = make_and_restore_model(arch='resnet50', dataset=ds, resume_path=resume_path)

    return model
