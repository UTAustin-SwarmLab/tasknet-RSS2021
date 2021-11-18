Goal: 
    Train and test basic CNNs in pytorch.

Example:
    run_convnet_MNIST.sh
    Saves pth of MNIST model and evaluates accuracy to 99%
    - calls train_task_convnet.py

    Reproducible result:
        - training for 5 epochs yields test accuracy of 99.16%

Utils:
    utils_load_datasets.py
        - data loaders for MNIST, other datasets

    ConvNet_utils.py
        - simple architectures for MNIST DNN

    utils_train_test.py
        - pytorch loops to train/test a DNN
