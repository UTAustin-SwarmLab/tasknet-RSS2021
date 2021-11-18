#!/bin/bash


#export CUDA_VISIBLE_DEVICES=0

# path to mars terrain data downloaded from the following link
# https://drive.google.com/file/d/1zu0y3exVJb6LoTwUodHNP3D4ESnOU-vQ/view?usp=sharing
DATA_PATH=path_to_data.npz

# directory where pretrained CNN weights downloaded from the following link are placed
# https://drive.google.com/drive/folders/1-4lg8ychyDesaNqMub3O4-eATulOyLsq?usp=sharing
PRETRAIN_WEIGHT_DIR=path_to_pretrain_weight

# output directory
OUT_DIR=out_dir

# Environment ID, terrain data contains 3 environments
ENV_ID=0

# Specify start/end point. they must be between 0 and 224 in the provided data
START_X=0
START_Y=0
END_X=224
END_Y=224

python plan_mars_terrain.py --data_path $DATA_PATH --weight_dir $PRETRAIN_WEIGHT_DIR --out_dir $OUT_DIR --env_id $ENV_ID --start_end $START_X $START_Y $END_X $END_Y
