
TRAIN_DATA=path_to_train_data
VAL_DATA=path_to_val_data

for Z_DIM in 1 2 3 4 5 6 8 12 16;

do
  PRETRAIN_AE_CKPT=path_to_ckpt.pt
  PRETRAIN_TASK_CKPT=path_to_ckpt.pt

	python train_tasknet.py --train_scheme eval_only --z_dim ${Z_DIM} --pretrain_ckpt $PRETRAIN_AE_CKPT $PRETRAIN_TASK_CKPT --data_dir $TRAIN_DATA --val_data_dir $VAL_DATA

done
