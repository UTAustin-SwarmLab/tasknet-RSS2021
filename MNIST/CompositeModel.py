import sys,os
import torch
from torch import nn
import numpy as np
from collections import OrderedDict
import torchvision

from convNet.ConvNet_utils import ConvNet
from VAE.model import VAE
from VAE.utils import load_checkpoint
from adversarial.utils_adversarial import load_robust_model

'''
    total parameters in a model
'''

def get_total_param_count(param_list = None, model = None):
    total_params = 0
    for param in param_list:
        #total_params += np.prod(param.data.numpy().shape)
        total_params += np.prod(param.cpu().data.numpy().shape)
    return total_params

'''
    how many total parameters are trainable?
'''
def get_total_trainable_params(model = None):
    total_param_list = [p for p in model.parameters()]
    trainable_param_list = [p for p in model.parameters() if p.requires_grad]

    total_param_count = get_total_param_count(param_list = total_param_list, model = model)
    trainable_param_count = get_total_param_count(param_list = trainable_param_list, model = model)

    return total_param_count, trainable_param_count

'''
    freeze weights for model, can't be updated in backprop
'''

def freeze_model_weights(model = None):
    total_params = 0
    for param in model.parameters():
        param.requires_grad = False
        #total_params += np.prod(param.data.numpy().shape)
        total_params += np.prod(param.cpu().data.numpy().shape)

    print('TOTAL PARAMS FROZEN: ', total_params)
    print(' ')
    return model

'''
    tasknet architecture
    has a VAE + CNN, VAE encodes and decodes then passes decoded image back to CNN
'''

class JointVAECNN(nn.Module):

    def __init__(self, label = None, image_size = None, channel_num = None, kernel_num = 128, z_size = None, which_model_pretrain = 'none', pretrain_CNN_path = None, cnn_type = 'MNIST', pretrain_VAE_path = None, use_robustness_lib = None, robustness_model_data_dir = None, num_encoder_layers=3):
        # configurations
        super().__init__()

        # declare autoencoder
        self.VAE = VAE(cnn_type, image_size, channel_num, kernel_num, z_size, enc_layer_num=num_encoder_layers)


        # declare CNN
        print('channel_num: ', channel_num)

        print('the CNN type is :', cnn_type)
        if cnn_type in ['MNIST', 'CIFAR10']:

            if not use_robustness_lib:
                self.CNN = ConvNet()
            else:
                self.CNN = load_robust_model(data_path=robustness_model_data_dir)
        else:
            pass

        # decide which (if any) pre-trained weights to use!
        # if use_pretrained, load the CNN weights from a file and set no grad for all parameters there
        if which_model_pretrain == 'only_CNN':
            print('STARTING TO LOAD THE CNN ONLY, robustness: ', use_robustness_lib)
            self.CNN = load_and_freeze_pretrain_CNN_params(self.CNN, pretrain_CNN_path, data_path = robustness_model_data_dir, use_robustness_lib = use_robustness_lib, dataset=cnn_type)

        elif which_model_pretrain == 'both_CNN_VAE':
            print('STARTING TO LOAD THE CNN (1/2), robustness: ', use_robustness_lib)
            self.CNN = load_and_freeze_pretrain_CNN_params(self.CNN, pretrain_CNN_path, data_path = robustness_model_data_dir, use_robustness_lib = use_robustness_lib, dataset=cnn_type)

            print('STARTING TO LOAD THE VAE (2/2)')
            _ = load_checkpoint(self.VAE, pretrain_VAE_path)
            self.VAE = freeze_model_weights(model = self.VAE)
        else:
            pass


        CNN_total_params, CNN_params_trainable = get_total_trainable_params(model = self.CNN)
        print('CNN params trainable: ', CNN_params_trainable, 'CNN total: ', CNN_total_params)
        print(' ')

        VAE_total_params, VAE_params_trainable = get_total_trainable_params(model = self.VAE)
        print('VAE params trainable: ', VAE_params_trainable, 'VAE total: ', VAE_total_params)
        print(' ')

    def forward(self, input):
        # first run VAE
        (VAE_mean, VAE_logvar), VAE_reconstructed = self.VAE(input)

        # then pass reconstruction as input to CNN
        CNN_output, _ = self.CNN(VAE_reconstructed)

        VAE_info = [VAE_mean, VAE_logvar, VAE_reconstructed]

        return CNN_output, VAE_info

'''
    once we have the trained integrated VAE, CNN, evaluate
    classification error and reconstruction error
'''

def evaluate_integrated_VAE_CNN(model, test_loader, dataset_info_dict, device, weights_must_be_loaded = True, PLOT_INTERVAL=10):

    # MAIN TEST FUNCTION
    ######################################################
    print('STARTING TO LOAD THE MODEL')

    if weights_must_be_loaded:
        checkpoint = torch.load(dataset_info_dict['model_save_ckpt'])
        model.load_state_dict(checkpoint)

    # Test the model
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        # classification stats
        correct = 0
        total = 0

        # reconstruction stats
        sum_reconstruction_loss = 0.0
        sum_kl_divergence_loss =  0.0
        sum_total_loss = 0.0

        for i, data_vec in enumerate(test_loader):

            images = data_vec[0]
            labels = data_vec[dataset_info_dict['task_index']]

            images = images.to(device)
            labels = labels.to(device)

            # whole end-to-end model
            # Forward pass: run the model
            outputs, VAE_info = model(images)

            # how good were the reconstructions?
            VAE_mean = VAE_info[0]
            VAE_logvar = VAE_info[1]
            VAE_reconstructed = VAE_info[2]

            reconstruction_loss = model.VAE.reconstruction_loss(VAE_reconstructed, images)
            kl_divergence_loss = model.VAE.kl_divergence_loss(VAE_mean, VAE_logvar)
            total_loss = reconstruction_loss + kl_divergence_loss

            # sum across the full dataset
            sum_reconstruction_loss += reconstruction_loss.item()
            sum_kl_divergence_loss += kl_divergence_loss.item()
            sum_total_loss += total_loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


            # plot selected images and their task-agnostic reconstructions
            if PLOT_INTERVAL > 0 and i % PLOT_INTERVAL == 0:
                plot_reconstructed_images(images, dataset_info_dict['batch_size'], i, VAE_reconstructed, dataset_info_dict['z_size'], dataset_info_dict['plot_dir'], channel_num = dataset_info_dict['channel_num'], pixel_dim_x = dataset_info_dict['image_size'], max_images=20)



        loss_info_dict = OrderedDict()
        loss_info_dict['correct'] = correct
        loss_info_dict['total'] = total
        loss_info_dict['sum_total_loss'] = sum_total_loss
        loss_info_dict['sum_kl_divergence_loss'] = sum_kl_divergence_loss
        loss_info_dict['sum_reconstruction_loss'] = sum_reconstruction_loss

        write_summary_results_VAE_CNN(loss_info_dict, dataset_info_dict)


def write_summary_results_VAE_CNN(loss_info_dict, dataset_info_dict):

    # write the results to a file
    results_fname = dataset_info_dict['results_subdir'] + '/' + '_'.join(['summary', dataset_info_dict['dataset_name'], dataset_info_dict['task'], dataset_info_dict['run_prefix'], 'zdim', str(dataset_info_dict['z_size'])]) + '.txt'

    with open(results_fname, 'w') as f:
        header_str = '\t'.join(['dataset_name', 'task', 'run_prefix', 'correct', 'total', 'accuracy', 'z_dim', 'num_points', 'avg_vae_loss', 'avg_kl_loss', 'avg_reconstruction_loss', 'experiment_type'])
        f.write(header_str + '\n')

        correct = loss_info_dict['correct']
        total = loss_info_dict['total']
        accuracy = round(100*loss_info_dict['correct'] / total, 4)
        avg_vae_loss = round(loss_info_dict['sum_total_loss']/total , 4)
        avg_kl_loss = round(loss_info_dict['sum_kl_divergence_loss']/total, 4)
        avg_reconstruction_loss = round(loss_info_dict['sum_reconstruction_loss']/total, 4)

        results_str = '\t'.join([dataset_info_dict['dataset_name'], dataset_info_dict['task'], dataset_info_dict['run_prefix'], str(correct), str(total), str(accuracy), str(dataset_info_dict['z_size']), str(total), str(avg_vae_loss), str(avg_kl_loss), str(avg_reconstruction_loss), str(dataset_info_dict['experiment_type'])])
        f.write(results_str + '\n')


def freeze_optimizer_weights(model, which_model_pretrain, learning_rate):
    # case end-to-end: all of the model parameters are free to be changed
    if which_model_pretrain == 'none':

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        num_weights_to_train_list = [1 for p in model.parameters() if p.requires_grad]
        print('NOT using pretrained weights, training all: ', sum(num_weights_to_train_list))

    # case task-aware or task-agnostic: we freeze certain weights
    else:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

        num_weights_to_train_list = [1 for p in model.parameters() if p.requires_grad]
        print('using pretrained weights, only training: ', sum(num_weights_to_train_list))

    print('MODE: ', which_model_pretrain)
    return optimizer

def freeze_optimizer_weights_for_fine_tuning(model, learning_rate):
    for p in model.VAE.parameters():
        p.requires_grad = False
    cnn_params = [p for p in model.CNN.parameters()]
    for p in cnn_params[:-2]:
        p.requires_grad = False
    for p in cnn_params[-2:]:
        p.requires_grad = True

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    num_weights_to_train_list = [1 for p in model.parameters() if p.requires_grad]
    print('fine-tuning CNN of tasknet, only training: ', sum(num_weights_to_train_list))
    return optimizer

'''
    different loading techniques for our CNNs vs robust models
'''
def load_and_freeze_pretrain_CNN_params(model, pretrain_CNN_path, data_path = None, arch='resnet50', dataset='MNIST', use_robustness_lib = False):

    if use_robustness_lib:
        CNN_model = load_robust_model(data_path=data_path, dataset=dataset, arch=arch, resume_path=pretrain_CNN_path)
    else:
        checkpoint = torch.load(pretrain_CNN_path)
        CNN_model.load_state_dict(checkpoint)
    CNN_model = freeze_model_weights(model = CNN_model)

    return CNN_model


def plot_reconstructed_images(images, batch_size, batch_num, VAE_reconstructed, z_dim, results_dir, channel_num = 3, pixel_dim_x = 32, max_images=20):
    n = min(images.size(0), max_images)
    min_batch = min(batch_size, VAE_reconstructed.shape[0])
    new_x = VAE_reconstructed.view(min_batch, channel_num, pixel_dim_x, pixel_dim_x)


    print(results_dir)

    comparison = torch.cat([images[:n], new_x[:n]])
    torchvision.utils.save_image(comparison.cpu(), results_dir + '/task_recon_zdim_' + str(z_dim) + '_reconstruction_' + str(batch_num) + '.png', nrow=n)
