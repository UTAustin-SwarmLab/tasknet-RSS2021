#Assumed following environ variables are set
# - CUDA_VISIBLE_DEVICES
# - RUN_PREFIX
# - PRETRAIN_CNN_PATH
# - CNN_ARCH

# what GPU does the code run on?
#GPU_NUM=0
#export CUDA_VISIBLE_DEVICES=${GPU_NUM}

DATASET='MNIST'
CNN_ARCH='resnet50'

# if we use a fixed CNN (pre-trained CNN)
WHICH_MODEL_PRETRAIN='only_CNN'

RUN_PREFIX='run_adv'

LAMBDA=0.01

# where results go
SCRATCH_DIR=scratch/JointTrain_${DATASET}_pretrain_${WHICH_MODEL_PRETRAIN}_${RUN_PREFIX}/
rm -rf $SCRATCH_DIR
mkdir -p $SCRATCH_DIR

# how long to train for
MAX_EPOCHS=10

DATA_DIR=data/

# where results go
TASK='digit'
PRETRAIN_CNN_PATH=pretrain/20200618-021047/checkpoint.pt.best
#PRETRAIN_CNN_PATH=pretrain/20200619-005230/checkpoint.pt.best

RESULTS_DIR=$SCRATCH_DIR/Joint_VAE_CNN_train_results
rm -rf ${RESULTS_DIR}
mkdir -p ${RESULTS_DIR}

for Z_DIM in 1 2 3 4 8 16 32;

do
  CKPT_DIR=$SCRATCH_DIR/ckpts_${Z_DIM}/
	# train the VAE and CNN jointly

  python3 tasknet_full.py --epochs $MAX_EPOCHS --ckpt-dir $CKPT_DIR --results-dir $RESULTS_DIR --dataset $DATASET --task $TASK --z-size $Z_DIM --run_prefix ${RUN_PREFIX} --train-mode --pretrain-CNN-path ${PRETRAIN_CNN_PATH} --which-model-pretrain ${WHICH_MODEL_PRETRAIN} --cnn_arch ${CNN_ARCH} --use-robustness-lib --vae-loss-fraction $LAMBDA

done
