import os

NUM_FEATURES = 4
SCENARIO_LIST = ['heat_shock', 'light_switch', 'tamper_sensor']
RESULT_CSV_COLUMNS = [
    'time',
    'net',
    'num_layers_ae',
    'num_layers_tasknet',
    'unit_size_tasknet',
    'z_dim',
    'window_size',
    'data_type',
    'val_fraction',
    'lr',
    'recon_loss_weight',
    'num_epochs',
    'saved_ckpt_path',
    'trained_net',
    'pretrain_ae',
    'pretrain_tasknet',
    'pretrain_jointnet',
    'reconstruction_loss',
    'accuracy'
    ]
