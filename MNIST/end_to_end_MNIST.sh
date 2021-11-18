# what GPU does the code run on?
GPU_NUM=3
export CUDA_VISIBLE_DEVICES=${GPU_NUM}

DATASET='MNIST'

# if we learn all weights end-to-end
WHICH_MODEL_PRETRAIN='none'

RUN_PREFIX=run3

# where results go
SCRATCH_DIR=scratch/JointTrain_${DATASET}_pretrain_${WHICH_MODEL_PRETRAIN}_${RUN_PREFIX}/
rm -rf $SCRATCH_DIR
mkdir -p $SCRATCH_DIR

# how long to train for
MAX_EPOCHS=10

DATA_DIR=data/

# where results go
TASK='digit'


RESULTS_DIR=$SCRATCH_DIR/Joint_VAE_CNN_train_results
rm -rf ${RESULTS_DIR}
mkdir -p ${RESULTS_DIR}

# loop over the size of the latent code Z
for Z_DIM in 1 2 3 4 8 16 32;

do
    CKPT_DIR=$SCRATCH_DIR/ckpts_${Z_DIM}/

	# train the VAE and CNN jointly
	python3 tasknet_full.py --epochs $MAX_EPOCHS --ckpt-dir $CKPT_DIR --results-dir $RESULTS_DIR --dataset $DATASET --task $TASK --z-size $Z_DIM --run_prefix ${RUN_PREFIX} --train-mode --which-model-pretrain ${WHICH_MODEL_PRETRAIN}

done
