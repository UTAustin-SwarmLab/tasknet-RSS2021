from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
import sys,os
from collections import OrderedDict
from CompositeModel import *

from textfile_utils import *
from convNet.utils_load_datasets import *
from convNet.utils_train_test import *

# arguments
parser = argparse.ArgumentParser(description='ConvNet')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--ckpt-dir', type=str, default='ckpt', metavar='C',
                    help='where to save model ckpts')
parser.add_argument('--results-dir', type=str, default='results/',
                    help='where to place images')
parser.add_argument('--dataset_name', type=str, default='MNIST', help='use MNIST or custom dataset')
parser.add_argument('--task', type=str, help='task')
parser.add_argument('--run_prefix', type=str, help='task')

parser.add_argument('--cnn_arch', type=str, help='cnn architecture')
parser.add_argument('--fine_tuning', action='store_true')

main_command = parser.add_mutually_exclusive_group(required=True)
main_command.add_argument('--train-mode', action='store_true', dest='train_mode')
main_command.add_argument('--test-mode', action='store_false')

parser.add_argument('--kernel-num', type=int, default=128)
parser.add_argument('--z-size', type=int, default=128)
parser.add_argument('--num-encoder-layers', type=int, default=3)

# which model do we pretrain?
# options: none (end-to-end), only_CNN (task-aware), both_CNN_VAE (task-agnostic)
parser.add_argument('--which-model-pretrain', type=str, default='CNN')
parser.add_argument('--pretrain-CNN-path', type=str)
parser.add_argument('--pretrain-VAE-path', type=str)
parser.add_argument('--robustness-model-data-dir', type=str, default=None)
parser.add_argument('--use-robustness-lib', action='store_true', default=False)
parser.add_argument('--vae-loss-fraction', type=float, default=0.0)

if __name__ == "__main__":

    # paste keyword args
    ##########################################
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Device configuration: GPU or CPU?
    print('args.cuda: ',  args.cuda)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")

    # Hyper parameters
    num_epochs = args.epochs
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

    # Load the CNN model, now we integrate with VAE
    ##########################################
    model = JointVAECNN(image_size = dataset_info_dict['image_size'] , channel_num = dataset_info_dict['channel_num'], kernel_num = args.kernel_num , z_size = args.z_size , which_model_pretrain = args.which_model_pretrain, pretrain_CNN_path = args.pretrain_CNN_path, cnn_type = args.dataset_name, pretrain_VAE_path = args.pretrain_VAE_path, use_robustness_lib = args.use_robustness_lib, robustness_model_data_dir = args.robustness_model_data_dir, num_encoder_layers=args.num_encoder_layers)

    model.to(device)

    # Loss and optimizer
    # loss is same - still want to compare task loss in counting space
    ##########################################
    criterion = nn.CrossEntropyLoss()

    # where do I save the trained model?
    ##########################################
    model_subdir = args.ckpt_dir + '/' + args.task + '/'
    model_save_ckpt = model_subdir + '/INTEGRATED_dataset_' + args.dataset_name + '_task_' + str(args.task) +  '_model.ckpt'
    plot_dir = args.results_dir + '/plots_' + str(args.z_size)
    remove_and_create_dir(plot_dir)

    # load key info about dataset
    dataset_info_dict['dataset_name'] = args.dataset_name
    dataset_info_dict['task'] = args.task
    dataset_info_dict['run_prefix'] = args.run_prefix
    dataset_info_dict['model_subdir'] = model_subdir
    dataset_info_dict['model_save_ckpt'] = model_save_ckpt
    dataset_info_dict['results_subdir'] = args.results_dir
    dataset_info_dict['z_size'] = args.z_size
    dataset_info_dict['experiment_type'] = args.which_model_pretrain
    dataset_info_dict['use_robustness_lib'] = args.use_robustness_lib
    dataset_info_dict['batch_size'] = args.batch_size
    dataset_info_dict['plot_dir'] = plot_dir

    # Train the model
    if args.train_mode:

        # optimizer is key part
        ##########################################
        optimizer = freeze_optimizer_weights(model, args.which_model_pretrain, learning_rate)
        train_taskDNN(model, train_loader, args.epochs, device, criterion, optimizer, dataset_info_dict, vae_loss_fraction=args.vae_loss_fraction)

        if args.fine_tuning:
            fine_tune_opt = freeze_optimizer_weights_for_fine_tuning(model, learning_rate)
            dataset_info_dict['model_save_ckpt'] = '.'.join(model_save_ckpt.split('.')[:-1]) + '_fine_tuned.ckpt'
            train_taskDNN(model, train_loader, 10, device, criterion, fine_tune_opt, dataset_info_dict, create_dir=False)
            dataset_info_dict['model_save_ckpt'] = model_save_ckpt

        # writes detailed loss info as well
        evaluate_integrated_VAE_CNN(model, test_loader, dataset_info_dict, device, weights_must_be_loaded = False)
    else:
        # writes detailed loss info as well
        if args.which_model_pretrain == 'both_CNN_VAE':
            weights_must_be_loaded = False
        else:
            weights_must_be_loaded = True

        evaluate_integrated_VAE_CNN(model, test_loader, dataset_info_dict, device, weights_must_be_loaded)
