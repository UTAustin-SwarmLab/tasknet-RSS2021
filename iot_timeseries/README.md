# TaskNet for IoT timeseries

## Install dependencies

`pip install -r requirements.txt`

## Pre-train model checkpoint

https://drive.google.com/file/d/1JbThOQxLIxzJyGTbmh3W6wzQ5vAgzZl6/view?usp=sharing

## Dataset

You can download timeseries data from the following link.

https://drive.google.com/drive/folders/1i3w46Rw7MvDz1cbxY4MBJ_bbP40FMu23?usp=sharing

Data is saved as multiple csv files whose directory paths are
*ROOT_DIR*/[train|val]/*SCENARIO_NAME*/*DATA#*/sensor_data_*TIMESTAMP*.csv

e.g. *ROOT_DIR*/train/tamper_sensor/1/sensor_data_time_15_07_2020-04:04:43.csv

## Running training and test

sample scripts are run_[ task_aware | task_agnostic | end_to_end ].sh

`python train_tasknet.py --help` shows more details of command options.
