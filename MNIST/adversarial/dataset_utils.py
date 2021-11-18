import os
import textfile_utils
import torch as ch
from torchvision import transforms, datasets
from robustness import datasets as rdatasets
from robustness import cifar_models
from robustness import data_augmentation as da
from robustness import defaults

data_path = os.path.join(textfile_utils.TASK_ROOT_DIR, 'data')

class MNIST(rdatasets.DataSet):

    def __init__(self, data_path=data_path, **kwargs):
        self.num_classes = 10
        ds_kwargs = {
            'num_classes': self.num_classes,
            'mean': ch.tensor([0.5, 0.5, 0.5]),
            'std': ch.tensor([0.3]),
            'custom_class': datasets.MNIST,
            'label_mapping': None,
            'transform_train': da.TRAIN_TRANSFORMS_DEFAULT(28),
            'transform_test': da.TEST_TRANSFORMS_DEFAULT(28)
        }
        super(MNIST, self).__init__('mnist', data_path, **ds_kwargs)

    def get_model(self, arch, pretrained):
        assert not pretrained, "pretrained only available for ImageNet"
        return cifar_models.__dict__[arch](num_classes=self.num_classes)

defaults.TRAINING_DEFAULTS[MNIST] = {
        "epochs": 20,
        "batch_size": 128,
        "weight_decay":5e-4,
        "step_lr": 50
    }

rdatasets.DATASETS['mnist'] = MNIST
