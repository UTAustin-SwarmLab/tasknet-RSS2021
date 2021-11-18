import sys,os
import torch

from textfile_utils import *
from adversarial.utils_adversarial import load_robust_model

"""
    train a deep network in pytorch
"""

def train_DNN(model, train_loader, num_epochs, device, criterion, optimizer, dataset_info_dict, print_mode = False, input_index=0):

    remove_and_create_dir(dataset_info_dict['model_subdir'])
    total_step = len(train_loader)
    print('Training with input index: ', input_index)
    for epoch in range(num_epochs):

        # 0: image, 1: num_trees, 2: num_patch, 3: rob_x, 4: rob_y, 5: goal_x, 6: goal_y
        for i, data_vec in enumerate(train_loader):
            images = data_vec[input_index]
            labels = data_vec[dataset_info_dict['task_index']]
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs, _ = model(images)

            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if print_mode:
                print('labels: ', labels.size())
                print('outputs: ', outputs.size())
                print('labels: ', labels)
                print('outputs: ', outputs)


            if (i+1) % 10 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                       .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # Save the model checkpoint
    torch.save(model.state_dict(), dataset_info_dict['model_save_ckpt'])

def train_taskDNN(model, train_loader, num_epochs, device, criterion, optimizer, dataset_info_dict, print_mode = False, input_index=0, vae_loss_fraction=0.0, create_dir=True):

    if create_dir:
        remove_and_create_dir(dataset_info_dict['model_subdir'])
    total_step = len(train_loader)
    print('Training with input index: ', input_index)
    for epoch in range(num_epochs):

        # 0: image, 1: num_trees, 2: num_patch, 3: rob_x, 4: rob_y, 5: goal_x, 6: goal_y
        for i, data_vec in enumerate(train_loader):
            images = data_vec[input_index]
            labels = data_vec[dataset_info_dict['task_index']]
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            (mean, logvar), x_reconstructed = model.VAE(images)
            reconstruction_loss = model.VAE.reconstruction_loss(x_reconstructed, images)
            kl_divergence_loss = model.VAE.kl_divergence_loss(mean, logvar)
            vae_loss = reconstruction_loss + kl_divergence_loss

            CNN_output, _ = model.CNN(x_reconstructed)
            cnn_loss = criterion(CNN_output, labels)

            loss = vae_loss_fraction*vae_loss + (1-vae_loss_fraction)*cnn_loss

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if print_mode:
                print('labels: ', labels.size())
                print('outputs: ', CNN_output.size())
                print('labels: ', labels)
                print('outputs: ', CNN_output)


            if (i+1) % 10 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                       .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # Save the model checkpoint
    torch.save(model.state_dict(), dataset_info_dict['model_save_ckpt'])

"""
    train CNN of tasknet
"""
def train_CNN_of_tasknet(model, vae, train_loader, num_epochs, device, criterion, optimizer, dataset_info_dict, print_mode = False, input_index=0):

    remove_and_create_dir(dataset_info_dict['model_subdir'])
    total_step = len(train_loader)
    print('Training with input index: ', input_index)
    for epoch in range(num_epochs):

        # 0: image, 1: num_trees, 2: num_patch, 3: rob_x, 4: rob_y, 5: goal_x, 6: goal_y
        for i, data_vec in enumerate(train_loader):
            images = data_vec[input_index]
            labels = data_vec[dataset_info_dict['task_index']]
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            (mean, logvar), reconstructed = vae(images)
            outputs, _ = model(reconstructed)

            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if print_mode:
                print('labels: ', labels.size())
                print('outputs: ', outputs.size())
                print('labels: ', labels)
                print('outputs: ', outputs)


            if (i+1) % 10 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                       .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # Save the model checkpoint
    torch.save(model.state_dict(), dataset_info_dict['model_save_ckpt'])


"""
    test a pre-trained deep network in pytorch
"""

def test_DNN(model, test_loader, device, dataset_info_dict, input_index=0):

    remove_and_create_dir(dataset_info_dict['results_subdir'])

    if dataset_info_dict['robustness_train_type'] == 'no_robustness':
        print('STARTING TO LOAD THE MODEL')
        checkpoint = torch.load(dataset_info_dict['model_save_ckpt'])
        model.load_state_dict(checkpoint)
    else:
        model = load_robust_model(data_path = dataset_info_dict['robust_model_path'], resume_path=dataset_info_dict['robust_model_path'], dataset=dataset_info_dict['dataset_name'])
        print('LOADED robust model: ', dataset_info_dict['robustness_train_type'])
        print('path: ', dataset_info_dict['robust_model_path'])
        model.to(device)
    # Test the model
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for i, data_vec in enumerate(test_loader):

            images = data_vec[input_index]
            labels = data_vec[dataset_info_dict['task_index']]

            images = images.to(device)
            labels = labels.to(device)
            outputs,_ = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            correct += (predicted == labels).sum().item()

            if i % 10 == 0:
                print('progress: ', float(i)/len(test_loader))

        accuracy = 100 * correct / total
        print('Test Accuracy of the model: {} %'.format(accuracy))

        write_summary_results(dataset_info_dict['results_subdir'], dataset_info_dict['dataset_name'], dataset_info_dict['task'], dataset_info_dict['run_prefix'], correct, total)


    return accuracy

def write_summary_results(results_subdir, dataset_name, task, run_prefix, correct, total):

    # write the results to a file
    results_fname = results_subdir + '/' + '_'.join(['summary', dataset_name, task, run_prefix]) + '.txt'

    with open(results_fname, 'w') as f:
        header_str = '\t'.join(['dataset_name', 'task', 'run_prefix', 'correct', 'total', 'accuracy'])
        f.write(header_str + '\n')

        accuracy = 100*correct / total

        results_str = '\t'.join([dataset_name, task, run_prefix, str(correct), str(total), str(accuracy)])
        f.write(results_str + '\n')
