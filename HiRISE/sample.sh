#!/bin/bash
#export CUDA_VISIBLE_DEVICES=0
DATA_PATH=path_to_data
# pretrained CNN weight
PRETRAIN_WEIGHT=path_to_pretrain_weight

# split layers
for split_layer_name in 'block2b_add' 'block3b_add' 'block4c_add' 'block5c_add' 'block6d_add'

do

for latent_dim in 1 2 4 8 16 32 64 128

do

OUT_PREFIX=path_to_saved_weight

#TASK AWARE
#python efn_utils.py --split_layer_name $split_layer_name --data_path $DATA_PATH --weight_path $PRETRAIN_WEIGHT --weight_type classifier --train_scheme task_aware --data_augmentation --latent_dim $latent_dim --num_classes 8 --epochs 10 --save_path "$OUT_PREFIX".h5 > "$OUT_PREFIX".log

#END to END
#python efn_utils.py --split_layer_name $split_layer_name --train_scheme end_to_end --data_path $DATA_PATH --save_path "$OUT_PREFIX".h5 --data_augmentation --epochs 10 --latent_dim $latent_dim --num_classes 8  > "$OUT_PREFIX".log

#TEST
python efn_utils.py --test --split_layer_name $split_layer_name --train_scheme task_aware --data_path $DATA_PATH --weight_path $PRETRAIN_WEIGHT --weight_type composite --latent_dim $latent_dim --num_classes 8 > "$OUT_PREFIX".log

done

done
