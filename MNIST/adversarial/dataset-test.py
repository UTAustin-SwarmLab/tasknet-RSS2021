import os,sys,unittest
from robustness.model_utils import make_and_restore_model

import adversarial.dataset_utils

class TestAction(unittest.TestCase):

    def test_mnist_class(self):
        print('mnist class test...')
        ds = utils.dataset_utils.MNIST()
        model, _ = make_and_restore_model(arch='resnet50', dataset=ds)
        print(model)

if __name__ == '__main__':
    print("### dataset test starts... ###")
    unittest.main()
