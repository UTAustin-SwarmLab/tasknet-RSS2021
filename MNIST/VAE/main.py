# this code was modified from online, need citation!!!

#!/usr/bin/env python3
import argparse
import torch
import torchvision
from model import VAE
from data import TRAIN_DATASETS, DATASET_CONFIGS
from train import train_model
import utils


parser = argparse.ArgumentParser('VAE PyTorch implementation')
parser.add_argument('--dataset', default='mnist',
                    choices=list(TRAIN_DATASETS.keys()))

parser.add_argument('--kernel-num', type=int, default=128)
parser.add_argument('--z-size', type=int, default=128)

parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--sample-size', type=int, default=64)
#parser.add_argument('--lr', type=float, default=3e-05)
parser.add_argument('--lr', type=float, default=6e-05)
parser.add_argument('--weight-decay', type=float, default=1e-06)

parser.add_argument('--loss-log-interval', type=int, default=100)
parser.add_argument('--image-log-interval', type=int, default=500)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
parser.add_argument('--sample-dir', type=str, default='./samples')
parser.add_argument('--no-gpus', action='store_false', dest='cuda')

main_command = parser.add_mutually_exclusive_group(required=True)
main_command.add_argument('--test', action='store_false', dest='train')
main_command.add_argument('--train', action='store_true')


if __name__ == '__main__':
    args = parser.parse_args()
    cuda = args.cuda and torch.cuda.is_available()
    #cuda = torch.cuda.is_available()
    dataset_config = DATASET_CONFIGS[args.dataset]
    dataset = TRAIN_DATASETS[args.dataset]

    dataset_name = args.dataset
    print('cuda: ', cuda)

    device = torch.device("cuda" if args.cuda else "cpu") 

    vae = VAE(
        label=args.dataset,
        image_size=dataset_config['size'],
        channel_num=dataset_config['channels'],
        kernel_num=args.kernel_num,
        z_size=args.z_size,
    ).to(device)
    
    print('image_log: ', args.image_log_interval)

    # run a test or a training process.
    if args.train:
        train_model(
            vae, dataset=dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            sample_size=args.sample_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            checkpoint_dir=args.checkpoint_dir,
            loss_log_interval=args.loss_log_interval,
            image_log_interval=args.image_log_interval,
            resume=args.resume,
            cuda=cuda,
	    results_dir = args.sample_dir,
	    pixel_dim_x = dataset_config['size'],
	    num_channels = dataset_config['channels'],
	    zdim = args.z_size,
            device = device,
	    dataset_name = dataset_name
        )
    else:

        # first have to LOAD the model
        epoch_start = utils.load_checkpoint(vae, args.checkpoint_dir)
    
        vae.to(device)

        images = vae.sample(args.sample_size)
        test_image_file = args.sample_dir + '/TEST_' + '_'.join([dataset_name, 'z', str(args.z_size)]) + '.png'
        torchvision.utils.save_image(images, test_image_file)
