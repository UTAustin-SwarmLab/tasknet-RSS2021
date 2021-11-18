GPU_NUM=0
export CUDA_VISIBLE_DEVICES=${GPU_NUM}

# output directory
RUN_PREFIX=run4

SCRATCH_DIR=scratch/MNIST_cnn_${RUN_PREFIX}
#rm -rf $SCRATCH_DIR
#mkdir -p $SCRATCH_DIR

# how long to train
MAX_EPOCHS=5

# where to store model
CKPT_DIR=$SCRATCH_DIR/ckpts/

DATASET='MNIST'

# where results go
RESULTS_DIR=$SCRATCH_DIR/CNN_train_results

for TASK in 'digit'
do

	# train the VAE
	#python3 train_task_convnet.py --epochs $MAX_EPOCHS --ckpt_dir $CKPT_DIR --results_dir $RESULTS_DIR --dataset_name $DATASET --train_mode --task $TASK

	# test and draw images
	python3 train_task_convnet.py --epochs $MAX_EPOCHS --ckpt_dir $CKPT_DIR --results_dir $RESULTS_DIR --dataset_name $DATASET --test_mode --task $TASK --run_prefix ${RUN_PREFIX}

done
