import sys,os,argparse
import numpy as np
import torch
from PIL import Image

sys.path.append(os.environ['MPNET_LIB'])
import Model.model as mpnet_model
import Model.AE.CAE_r3d as CAE_r3d
import Model.AE.CAE as CAE_2d
#import data_loader_2d
import plan_s2d, plan_r3d
import utility_s2d, utility_r3d

from main import TaskMPNet
from utils.visualize_utils import plot_path
from utils.planning_utils_simple import run_MPNET

####### MAIN ########

def main(args):
    #load model
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('[info] cuda.set_device')
    else:
        device = torch.device('cpu')

    # setup evaluation function for s2d
    #IsInCollision = plan_s2d.IsInCollision
    normalize_s2d = utility_s2d.normalize
    unnormalize_s2d = utility_s2d.unnormalize
    CAE = CAE_2d
    MLP = mpnet_model.MLP

    # setup model
    task_mpnet = TaskMPNet(total_input_size=2804, AE_input_size=2800, mlp_input_size=28+4, output_size=2, AEtype='deep', n_tasks=1, n_memories=256, memory_strength=0.5, grad_step=1, CAE=CAE, MLP=MLP, latent_dim=4, get_z=False)
    task_mpnet.encoder.load_state_dict(torch.load(os.path.join(args.weight_dir, 'cae_encoder.pkl')))
    task_mpnet.mlp.load_state_dict(torch.load(os.path.join(args.weight_dir, 'mlp_100_4000_PReLU_ae_dd150.pkl')))
    task_mpnet.bottleneck_encoder.load_state_dict(torch.load(os.path.join(args.weight_dir, 'bottleneck_encoder.pt')))
    task_mpnet.bottleneck_decoder.load_state_dict(torch.load(os.path.join(args.weight_dir, 'bottleneck_decoder.pt')))

    task_mpnet.to(device)
    print('[info] task mpnet loaded')

    #load data
    dataset = np.load(args.data_path)
    image  = dataset['image'][args.env_id]
    obs = dataset['point_cloud'][args.env_id]
    world_size = image.shape[0]/2
    def normalize_func_s2d(x):
        return (x - world_size) / 20
        #return normalize_s2d(x, world_size)
    def unnormalize_func_s2d(x):
        return x * 20 + world_size
        #return unnormalize_s2d(x, world_size)
    def IsInCollision(x, obc):
        if np.any(x < 0): return True
        if np.any(x > world_size*2): return True
        return np.min([np.linalg.norm(x-ob) for ob in obc]) < 5
    def get_coordinate(obc, xmin, xmax, ymin, ymax):
        ret = (np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax))
        while np.min([np.linalg.norm(ret-ob) for ob in obc]) < 10:
            ret = (np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax))
        return ret
    print('[info] obstacle dataset loaded')

    #motion-planning according to given start/end locations
    planned_path = []
    planned_path_xy = []
    feasibility_vec = []
    print('[info] start planning env {}'.format(args.env_id))
    start_end_tuple = (torch.FloatTensor([args.start_end[0],args.start_end[1]]),
                            torch.FloatTensor([args.start_end[2],args.start_end[3]]))
    print('[info] start/end point {}'.format(start_end_tuple))
    fp, path, path_xy = run_MPNET(start_end_tuple, task_mpnet, obs, torch.FloatTensor(obs.flatten()), IsInCollision, normalize_func_s2d, unnormalize_func_s2d)
    print('[info] feasibility {}'.format(fp))
    print('[info] path {}'.format(path))
    planned_path.append(path)
    planned_path_xy.append(path_xy)
    feasibility_vec.append(fp)

    #visualize paths
    out_path = os.path.join(args.out_dir, 'path_env{}.png'.format(args.env_id))
    plot_path(obs, planned_path_xy, out_path, xlim=[0,world_size*2], ylim=[0,world_size*2], title_str='Env {}'.format(args.env_id), terrain_img=Image.fromarray(image))
    print('[info] path visualized to {}'.format(out_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='path to input npz file that contains image and obc')
    parser.add_argument('--weight_dir', type=str, help='path to trained weight')
    parser.add_argument('--out_dir', type=str, help='directory path where result image will be placed')
    parser.add_argument('--env_id', type=int)
    parser.add_argument('--start_end', nargs='+', type=int, help='coordinate for start/end point: [start_x, start_y, end_x, end_y]')

    args = parser.parse_args()
    print('start')
    main(args)
    print('done')
