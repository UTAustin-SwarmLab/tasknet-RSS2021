import os,sys
import argparse
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from const import *
from utils.model_utils import JointAETasknetFc
from utils.dataset_utils import *

def get_current_date_time(print_mode = False):

    # datetime object containing current date and time
    now = datetime.now()

    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d_%m_%Y-%H:%M:%S")

    if print_mode:
        print("now =", now)
        print("date and time =", dt_string)

    return dt_string

if __name__ == '__main__':

    start_time = get_current_date_time()
    print('[info] start training jointnet at {}'.format(start_time))

    parser = argparse.ArgumentParser(description='train joint network consists of autoencoder and tasknet')

    parser.add_argument('--scenario', type=str, choices=['all']+SCENARIO_LIST, default='all')
    parser.add_argument('--data_dir', type=str, help='path to training data')
    parser.add_argument('--val_data_dir', type=str, help='path to data directory of validation, if specified, val_fraction option will be ignored')
    parser.add_argument('--num_layers', nargs='+', default=[2, 3], help='number of layers: [autoendocer, tasknet]')
    parser.add_argument('--z_dim', type=int, default=4)
    parser.add_argument('--window_size', type=int, default=5)
    parser.add_argument('--unit_size', type=int, default=5, help='unit size of fully connected layers of tasknet')
    parser.add_argument('--val_fraction', type=float, default=0.1, help='fraction of data used for validation')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--recon_loss_weight', type=float, default=0, help='lambda')
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--train_scheme', type=str, choices=['end_to_end', 'autoencoder', 'tasknet', 'eval_only'], default='autoencoder')
    parser.add_argument('--pretrain_ckpt', nargs='+', help='path to pre-trained ckpt that should be consistent to train_scheme option')
    parser.add_argument('--ckpt_path', type=str, help='if specified, trained model weights will be saved on the path')
    parser.add_argument('--csv_path', type=str, help='path to result csv')

    args = parser.parse_args()

    jointnet = JointAETasknetFc(num_features = args.window_size*NUM_FEATURES,
                                num_layers = args.num_layers,
                                tasknet_unit_size = args.unit_size,
                                latent_dim = args.z_dim)

    print('[jointnet] num_features {}, z_dim {}, num_layers {}'.format(jointnet.num_features, jointnet.latent_dim, jointnet.num_layers))

    #data loading
    scenario_enabled = [args.scenario] if args.scenario != 'all' else SCENARIO_LIST
    train_loader, val_loader = get_dataloaders(scenario_enabled, args.data_dir, args.window_size, args.val_fraction, args.val_data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.BCELoss()
    ae_loss_fn = nn.MSELoss()

    best_accuracy = 0
    recon_loss_on_best_acc = None
    best_state_dict = None

    pretrain_ae, pretrain_tasknet, pretrain_jointnet = None, None, None
    if args.train_scheme == 'autoencoder':
        print('[info] train autoencoder. tasknet will be frozen')
        #load tasknet's weights
        pretrain_tasknet = args.pretrain_ckpt[0]
        jointnet.tasknet.load_state_dict(torch.load(pretrain_tasknet))
        #freeze weights
        for p in jointnet.tasknet.parameters():
            p.requires_grad = False
    elif args.train_scheme == 'tasknet':
        print('[info] train tasknet. autoencoder will be frozen')
        #load tasknet's weights
        pretrain_ae = args.pretrain_ckpt[0]
        jointnet.autoencoder.load_state_dict(torch.load(pretrain_ae))
        #freeze weights
        for p in jointnet.autoencoder.parameters():
            p.requires_grad = False
    elif args.train_scheme == 'eval_only':
        print('[info] evalate using pretrain weights')
        if len(args.pretrain_ckpt) == 1:
            pretrain_jointnet = args.pretrain_jointnet[0]
            jointnet.load_state_dict(torch.load(pretrain_jointnet))
        else:
            pretrain_ae = args.pretrain_ckpt[0]
            pretrain_tasknet = args.pretrain_ckpt[1]
            jointnet.autoencoder.load_state_dict(torch.load(pretrain_ae))
            jointnet.tasknet.load_state_dict(torch.load(pretrain_tasknet))
    else:
        assert args.train_scheme == 'end_to_end'
        print('[info] train end to end')

    if args.train_scheme == 'eval_only':
        print('[info] start evaluation')
        jointnet.to(device)
        jointnet.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            reconstruction_loss = 0
            for i, data_vec in enumerate(val_loader):

                feature = data_vec[0].float()
                labels = data_vec[1].float()
                #print(labels)

                feature = feature.to(device)
                labels = labels.to(device)
                outputs, reconstructed = jointnet(feature)
                reconstruction_loss += ae_loss_fn(reconstructed, feature)
                pred = torch.argmax(outputs, 1)
                true_label = torch.argmax(labels, 1)
                total += labels.size(0)

                correct += (pred == true_label).sum().item()

            accuracy = 100 * correct / total
            best_accuracy = accuracy
            recon_loss_on_best_acc = reconstruction_loss.item()/len(val_loader)
        print ('Test Recon Loss: {}, Test acc: {}%'.format(round(reconstruction_loss.item()/len(val_loader), 4), round(accuracy, 2)))

    else:
        #train jointnet
        optimizer = torch.optim.SGD(jointnet.parameters(), args.lr, momentum=0.9)

        print('[info] start training')
        jointnet.to(device)
        for epoch in range(args.num_epochs):
            jointnet.train()

            for i, data_vec in enumerate(train_loader):
                feature = data_vec[0].float()
                labels = data_vec[1].float()
                feature = feature.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs, reconstructed = jointnet(feature)
                loss = criterion(outputs, labels) + args.recon_loss_weight * ae_loss_fn(reconstructed, feature)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                # gradient clipping
                torch.nn.utils.clip_grad_norm_(jointnet.parameters(), 1)
                optimizer.step()

            jointnet.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                reconstruction_loss = 0
                for i, data_vec in enumerate(val_loader):

                    feature = data_vec[0].float()
                    labels = data_vec[1].float()
                    #print(labels)

                    feature = feature.to(device)
                    labels = labels.to(device)
                    outputs, reconstructed = jointnet(feature)
                    reconstruction_loss += ae_loss_fn(reconstructed, feature)
                    pred = torch.argmax(outputs, 1)
                    true_label = torch.argmax(labels, 1)
                    total += labels.size(0)

                    correct += (pred == true_label).sum().item()

                accuracy = 100 * correct / total
            print ('Epoch [{}/{}], Train Loss: {:.4f}, Test Recon Loss: {}, Test acc: {}%'.format(epoch+1, args.num_epochs, loss.item(), round(reconstruction_loss.item()/len(val_loader), 4), round(accuracy, 2)))

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                recon_loss_on_best_acc = reconstruction_loss.item()/len(val_loader)
                best_state_dict = jointnet.state_dict()

        # Save the model checkpoint
        if args.ckpt_path is not None:
            torch.save(best_state_dict, args.ckpt_path)
            print('[info] checkpoint saved as {}'.format(args.ckpt_path))

    # Save the result to csv
    if args.result_csv is not None:
        result = {
            'time': start_time ,
            'net': 'jointnet',
            'num_layers_ae': jointnet.autoencoder.num_layers,
            'num_layers_tasknet': jointnet.tasknet.num_layers,
            'unit_size_tasknet': jointnet.tasknet.unit_size,
            'z_dim': args.z_dim,
            'window_size': args.window_size,
            'data_type': args.scenario,
            'val_fraction': args.val_fraction,
            'lr': args.lr,
            'recon_loss_weight': args.recon_loss_weight,
            'num_epochs': args.num_epochs,
            'saved_ckpt_path': args.ckpt_path,
            'trained_net': args.train_scheme,
            'pretrain_ae': pretrain_ae,
            'pretrain_tasknet': pretrain_tasknet,
            'pretrain_jointnet': pretrain_jointnet,
            'reconstruction_loss': recon_loss_on_best_acc,
            'accuracy': best_accuracy}
        assert set(result.keys()).issubset(set(RESULT_CSV_COLUMNS))

        df = pd.read_csv(args.csv_path, header=0) if os.path.exists(args.csv_path) else pd.DataFrame(columns=RESULT_CSV_COLUMNS)
        df = df.append(result, ignore_index=True)
        df.to_csv(args.csv_path, header=RESULT_CSV_COLUMNS, index=False)
        print('[info] saved result to csv file {}'.format(args.csv_path))

    print('[info] done')
