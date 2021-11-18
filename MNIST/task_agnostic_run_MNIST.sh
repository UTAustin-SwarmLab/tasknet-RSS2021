# this assumes a pre-trained CNN AND pre-trained VAE
# simply evaluate a composite, pre-trained model for various Z_DIM

# what GPU does the code run on?
GPU_NUM=0
export CUDA_VISIBLE_DEVICES=${GPU_NUM}

DATASET='MNIST'

# if we use a fixed CNN (pre-trained CNN)
WHICH_MODEL_PRETRAIN='both_CNN_VAE'

RUN_PREFIX=run3

# where results go
SCRATCH_DIR=scratch/JointTrain_${DATASET}_pretrain_${WHICH_MODEL_PRETRAIN}_${RUN_PREFIX}/
rm -rf $SCRATCH_DIR
mkdir -p $SCRATCH_DIR

DATA_DIR=data/

# where results go
TASK='digit'

PRETRAIN_CNN_PATH=scratch/MNIST_cnn_${RUN_PREFIX}/ckpts/digit/dataset_${DATASET}_task_${TASK}_model.ckpt

RESULTS_DIR=$SCRATCH_DIR/Joint_VAE_CNN_train_results
rm -rf ${RESULTS_DIR}
mkdir -p ${RESULTS_DIR}

BASE_VAE_DIR=scratch/backup_MNIST/MNIST_VAE_${RUN_PREFIX}/

for Z_DIM in 1 2 3 4 8 16 32;

do

    PRETRAIN_VAE_PATH=${BASE_VAE_DIR}/checkpoint_${DATASET}_${Z_DIM}

    CKPT_DIR=$SCRATCH_DIR/ckpts_${Z_DIM}/
	# train the VAE and CNN jointly
	python3 tasknet_full.py --ckpt-dir $CKPT_DIR --results-dir $RESULTS_DIR --dataset $DATASET --task $TASK --z-size $Z_DIM --run_prefix ${RUN_PREFIX} --pretrain-CNN-path ${PRETRAIN_CNN_PATH} --pretrain-VAE-path ${PRETRAIN_VAE_PATH} --which-model-pretrain ${WHICH_MODEL_PRETRAIN} --test-mode --use-robustness-lib
done
