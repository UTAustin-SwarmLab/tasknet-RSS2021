from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np

class TaskNetFc(nn.Module):
    def __init__(self, num_features=5*4, num_layers=3, unit_size=5, num_classes=2):
        assert num_layers > 1
        super(TaskNetFc, self).__init__()
        self.num_features = num_features
        self.num_layers = num_layers
        self.unit_size = unit_size
        self.num_classes = num_classes

        self.fc_first = nn.Linear(num_features, unit_size)
        self.fc_last = nn.Linear(unit_size, num_classes)
        hidden_layers_dict = OrderedDict()
        for i in range(num_layers-2):
            hidden_layers_dict['fc_'+str(i+1)] = nn.Linear(unit_size, unit_size)
        self.hidden_layers = nn.Sequential(hidden_layers_dict)

        self.sm = nn.Softmax(1)

    def forward(self, x):
        out = self.fc_first(x)
        out = self.hidden_layers(out)
        out = self.fc_last(out)
        out = self.sm(out)
        return out

class AutoEncoderFc(nn.Module):
    def __init__(self, num_features=5*4, num_layers=2, latent_dim=4):
        super(AutoEncoderFc, self).__init__()
        self.num_features = num_features
        self.num_layers = num_layers
        self.latent_dim = latent_dim

        enc_dict = OrderedDict([('input', nn.Linear(num_features, latent_dim*2**(num_layers-1)))])
        for i in range(num_layers-1):
            enc_dict['enc_relu'+str(i)] = nn.ReLU()
            enc_dict['enc'+str(i+1)] = nn.Linear(latent_dim*2**(num_layers-i-1), latent_dim*2**(num_layers-i-2))
        self.encoder = nn.Sequential(enc_dict)

        dec_dict = OrderedDict()
        for i in range(num_layers-1):
            dec_dict['dec'+str(i+1)] = nn.Linear(latent_dim*2**i, latent_dim*2**(i+1))
            dec_dict['dec_relu'+str(i)] = nn.ReLU()
        dec_dict['output'] = nn.Linear(latent_dim*2**(num_layers-1), num_features)
        self.decoder = nn.Sequential(dec_dict)

    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed

class JointAETasknetFc(nn.Module):
    def __init__(self, num_features=5*4, num_layers=(2,3), tasknet_unit_size=10, latent_dim=4, num_classes=2):
        super(JointAETasknetFc, self).__init__()
        self.num_features = num_features
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.tasknet_unit_size = tasknet_unit_size

        self.autoencoder = AutoEncoderFc(num_features, num_layers[0], latent_dim)
        self.tasknet = TaskNetFc(num_features, num_layers[1], tasknet_unit_size, num_classes)

    def forward(self, x):
        reconstructed = self.autoencoder(x)
        out = self.tasknet(reconstructed)
        return out, reconstructed
