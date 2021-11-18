import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms
import sys,os
from ConvNet_utils import ConvNet

from textfile_utils import *
from convNet.utils_load_datasets import *
from convNet.utils_train_test import *

parser = argparse.ArgumentParser(description='ConvNet')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=60, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--ckpt_dir', type=str, default='ckpt', metavar='C',
                    help='where to save model ckpts')
parser.add_argument('--results_dir', type=str, default='results/',
                    help='where to place images')
parser.add_argument('--dataset_name', type=str, default='MNIST', help='use MNIST or custom dataset')
parser.add_argument('--task', type=str, help='task')
parser.add_argument('--run_prefix', type=str, help='task')

# whether we use robust model
parser.add_argument('--robustness_train_type', type=str)
parser.add_argument('--robust_model_path', type=str)

main_command = parser.add_mutually_exclusive_group(required=True)
main_command.add_argument('--train_mode', action='store_true', dest='train_mode')
main_command.add_argument('--test_mode', action='store_false')

if __name__ == "__main__":

    # paste keyword args
    ##########################################
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Device configuration
    print('args.cuda: ',  args.cuda)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")

    # Hyper parameters
    learning_rate = 0.001

    # load the datasets
    ##########################################
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    # load train and test data for MNIST
    ##########################################
    if args.dataset_name == 'MNIST':
        # training dataset
        train_loader, test_loader, dataset_info_dict = load_MNIST_data(args.batch_size, kwargs)

    elif args.dataset_name == 'CIFAR10':
        train_loader, test_loader, dataset_info_dict = load_CIFAR10_data(args.batch_size, kwargs)

    # instantiate the model
    model = ConvNet().to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # where to save results
    model_subdir = args.ckpt_dir + '/' + args.task + '/'
    model_save_ckpt = model_subdir + '/dataset_' + args.dataset_name + '_task_' + str(args.task) +  '_model.ckpt'

    # load key info about dataset
    dataset_info_dict['dataset_name'] = args.dataset_name
    dataset_info_dict['task'] = args.task
    dataset_info_dict['run_prefix'] = args.run_prefix

    dataset_info_dict['model_subdir'] = model_subdir
    dataset_info_dict['model_save_ckpt'] = model_save_ckpt
    dataset_info_dict['results_subdir'] = args.results_dir
    dataset_info_dict['robustness_train_type'] = args.robustness_train_type
    dataset_info_dict['robust_model_path'] = args.robust_model_path

    # Train the model
    if args.train_mode:
        train_DNN(model, train_loader, args.epochs, device, criterion, optimizer, dataset_info_dict)
    # test a pre-trained model
    else:
        test_DNN(model, test_loader, device, dataset_info_dict)
