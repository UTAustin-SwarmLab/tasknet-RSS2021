TRAIN_DATA=path_to_train_data
VAL_DATA=path_to_val_data

for Z_DIM in 1 2 3 4 5 6 8 12 16;

do
	CKPT_PATH=path_to_checkpoint.pt
	python train_tasknet.py --train_scheme end_to_end --z_dim ${Z_DIM} --ckpt_path $CKPT_PATH --data_dir $TRAIN_DATA --val_data_dir $VAL_DATA

done
