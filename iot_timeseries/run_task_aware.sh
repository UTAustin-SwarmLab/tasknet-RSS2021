
PRETRAIN_TASK_CKPT=path_to_ckpt
TRAIN_DATA=path_to_train_data
VAL_DATA=path_to_val_data

for Z_DIM in 1 2 3 4 5 6 8 12 16;

do

	CKPT_PATH=path_to_saved_ckpt

	python train_tasknet.py --train_scheme autoencoder --z_dim ${Z_DIM} --pretrain_ckpt $PRETRAIN_TASK_CKPT --ckpt_path $CKPT_PATH --num_epochs 50 --recon_loss_weight 1 --data_dir $TRAIN_DATA --val_data_dir $VAL_DATA

done
