import os,glob
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

#metric: (mean, std)
norm_stats = {
    'ambient_light': (278, 231),
    'humidity': (31.8, 26.6),
    'pressure': (101, 0.0823),
    'temperature': (42.3, 15,4),
}

random_seed = 1024

def merge_dataset(datasets, window_size=5):
    data_dict = dict(light=[],humidity=[],pressure=[],temperature=[],datetime=[],anomaly=[])
    for dataset in datasets:
        data_dict['light'].extend(dataset.data['light'])
        data_dict['humidity'].extend(dataset.data['humidity'])
        data_dict['pressure'].extend(dataset.data['pressure'])
        data_dict['temperature'].extend(dataset.data['temperature'])
        data_dict['datetime'].extend(dataset.data['datetime'])
        data_dict['anomaly'].extend(dataset.data['anomaly'])
    return SensorDataset(window_size=window_size, data_dict=data_dict)

def split_dataset_into_train_and_val(dataset, val_frac):
    shuffled = shuffle(dataset.data['light'], dataset.data['humidity'], dataset.data['pressure'], dataset.data['temperature'], dataset.data['datetime'], dataset.data['anomaly'], random_state = random_seed)
    num_val = round(dataset.__len__()*val_frac)
    train_data_dict = dict(light=shuffled[0][:-num_val],
                           humidity=shuffled[1][:-num_val],
                           pressure=shuffled[2][:-num_val],
                           temperature=shuffled[3][:-num_val],
                           datetime=shuffled[4][:-num_val],
                           anomaly=shuffled[5][:-num_val])
    train_dataset = SensorDataset(window_size=dataset.window_size, data_dict=train_data_dict)
    val_data_dict = dict(light=shuffled[0][-num_val:],
                         humidity=shuffled[1][-num_val:],
                         pressure=shuffled[2][-num_val:],
                         temperature=shuffled[3][-num_val:],
                         datetime=shuffled[4][-num_val:],
                         anomaly=shuffled[5][-num_val:])
    val_dataset = SensorDataset(window_size=dataset.window_size, data_dict=val_data_dict)
    return train_dataset, val_dataset

#### dataset classes ####

class SensorDataset(Dataset):
    def __init__(self, file_path_list=None, window_size=5, normalize=True, data_dict=None):
        self.window_size = window_size
        if data_dict is not None:
            self.data = data_dict
        else:
            self.data = dict(light=[],humidity=[],pressure=[],temperature=[],datetime=[],anomaly=[])
            for file_path in file_path_list:
                dframe = self._normalize(pd.read_csv(file_path)) if normalize else pd.read_csv(file_path)
                for i in range(0, len(dframe)-window_size+1):
                    self.data['light'].append(list(dframe['ambient_light'][i:i+window_size]))
                    self.data['humidity'].append(list(dframe['humidity'][i:i+window_size]))
                    self.data['pressure'].append(list(dframe['pressure'][i:i+window_size]))
                    self.data['temperature'].append(list(dframe['temperature'][i:i+window_size]))
                    self.data['datetime'].append(dframe['current_datetime'][i+window_size-1])
                    self.data['anomaly'].append(self._is_anomaly(dframe['time_index'][i+window_size-1]))

    def _is_anomaly(self, time_index):
        raise Exception('anomaly classifier not implemented')

    def _normalize(self, df):
        for col in ['ambient_light', 'humidity', 'pressure', 'temperature']:
            df[col] = (df[col] - norm_stats[col][0]) / norm_stats[col][1]
        return df

    def __len__(self):
        return len(self.data['datetime'])

    def __getitem__(self, idx):
        feature = np.hstack([
            self.data['light'][idx],
            self.data['humidity'][idx],
            self.data['pressure'][idx],
            self.data['temperature'][idx]
        ])
        label = self.data['anomaly'][idx]
        return feature, label

class HeatShockDataset(SensorDataset):
    def __init__(self, file_path_list, window_size=5):
        super(HeatShockDataset, self).__init__(file_path_list, window_size)

    def _is_anomaly(self, time_index):
        if time_index < 20: return torch.tensor([0,1])
        if time_index < 50: return torch.tensor([1,0])
        if time_index < 70: return torch.tensor([0,1])
        return torch.tensor([1,0])

class LightSwitchDataset(SensorDataset):
    def __init__(self, file_path_list, window_size=5):
        super(LightSwitchDataset, self).__init__(file_path_list, window_size)

    def _is_anomaly(self, time_index):
        return torch.tensor([1,0])

class TamperSensorDataset(SensorDataset):
    def __init__(self, file_path_list, window_size=5):
        super(TamperSensorDataset, self).__init__(file_path_list, window_size)

    def _is_anomaly(self, time_index):
        if time_index < 20: return torch.tensor([0,1])
        if time_index < 40: return torch.tensor([1,0])
        if time_index < 60: return torch.tensor([0,1])
        if time_index < 80: return torch.tensor([1,0])
        return torch.tensor([0,1])


####### dataloaer utils #########

dataset_dict = {
    'heat_shock': HeatShockDataset,
    'light_switch': LightSwitchDataset,
    'tamper_sensor': TamperSensorDataset,
}

#load csv files and split into training and test
def get_dataloaders(scenarios, data_dir, window_size, val_fraction=0, val_data_dir=None):
    def gen_dataset(base_dir):
        csv_dict = {scenario: glob.glob(os.path.join(base_dir, scenario, '**', '*.csv'), recursive=True) for scenario in scenarios}
        datasets = [dataset_dict[scenario](csv_dict[scenario], window_size) for scenario in scenarios]
        return merge_dataset(datasets, window_size)
    if val_data_dir is not None:
        train_dataset, val_dataset = gen_dataset(data_dir), gen_dataset(val_data_dir)
    elif val_fraction > 0:
        train_dataset, val_dataset = split_dataset_into_train_and_val(gen_dataset(data_dir), val_fraction)
    else:
        print('[ERR] validation option must be specified. val_fraction {}, val_data_dir {}'.format(val_fraction, val_data_dir))
    print('[info] dataset loaded: {} training data / {} validation data'.format(train_dataset.__len__(), val_dataset.__len__()))

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
    return train_loader, val_loader
