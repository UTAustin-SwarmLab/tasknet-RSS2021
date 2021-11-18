import sys,os,argparse,pickle
import numpy as np
import torch.nn as nn
import torch
from torch import optim
from torch.multiprocessing import Pool, set_start_method

sys.path.append(os.environ['MPNET_LIB'])
from Model.GEM_end2end_model import End2EndMPNet
from gem_eval import eval_tasks
import Model.model as model
import Model.AE.CAE_r3d as CAE_r3d
import Model.AE.CAE as CAE_2d
import plan_s2d, plan_r3d
import data_loader_2d, data_loader_r3d
import utility_s2d, utility_r3d

from utils.planning_utils_simple import run_MPNET

######## tasknet class ##########

class TaskMPNet(End2EndMPNet):

    def __init__(self, total_input_size, AE_input_size, mlp_input_size, output_size, AEtype, n_tasks, n_memories, memory_strength, grad_step, CAE, MLP, latent_dim=None, get_z=False):
        super(TaskMPNet, self).__init__(total_input_size, AE_input_size, mlp_input_size, output_size, AEtype, n_tasks, n_memories, memory_strength, grad_step, CAE, MLP)
        self.latent_dim = latent_dim
        self.get_z = get_z
        if latent_dim is not None:
            self.bottleneck_encoder = nn.Sequential(nn.Linear(28, latent_dim))
            self.bottleneck_decoder = nn.Sequential(nn.Linear(latent_dim, 28))
        else:
            self.bottleneck_encoder = None
            self.bottleneck_decoder = None

    def forward(self, x):
        z = self.encoder(x[:,:self.AE_input_size])
        if self.bottleneck_encoder is not None:
            encoded = self.bottleneck_encoder(z)
            decoded = self.bottleneck_decoder(encoded)
        else:
            encoded, decoded = z, z

        mlp_in = torch.cat((decoded,x[:,self.AE_input_size:]), 1)
        if self.get_z:
            return self.mlp(mlp_in), z, decoded
        else:
            return self.mlp(mlp_in)

    def freeze_mpnet_params(self, target='all'):
        if target in ['all', 'encoder']:
            for param in self.encoder.parameters():
                param.requires_grad = False
        if target in ['all', 'mlp']:
            for param in self.mlp.parameters():
                param.requires_grad = False
        print('[info] {} of original mpnet parameters freezed'.format(target))

###### Dataset ########

class S2DDataset(torch.utils.data.Dataset):
    def __init__(self, obc, obs, paths, path_lengths):
        self.obc = obc
        self.obs = obs
        self.paths = paths
        self.path_lengths = path_lengths

        self._setup_index()
        assert np.sum(self.path_lengths) - np.count_nonzero(self.path_lengths) == len(self.global_idx)

    def _setup_index(self):
        self.global_idx = []
        for env_idx in range(self.path_lengths.shape[0]):
            for path_idx in range(self.path_lengths.shape[1]):
                for stride_idx in range(self.path_lengths[env_idx][path_idx]):
                    if stride_idx == self.path_lengths[env_idx][path_idx] - 1: break
                    self.global_idx.append(dict(env_idx=env_idx, path_idx=path_idx, stride_idx=stride_idx))

    def get_idx(self, idx):
        return self.global_idx[idx]['env_idx'], self.global_idx[idx]['path_idx'], self.global_idx[idx]['stride_idx']

    def __len__(self):
        return len(self.global_idx)

    def __getitem__(self, idx):
        env_idx, path_idx, stride_idx = self.get_idx(idx)
        obstacles = self.obs[env_idx]
        current_point = self.paths[env_idx][path_idx][stride_idx]
        next_target = self.paths[env_idx][path_idx][stride_idx+1]
        goal_point = self.paths[env_idx][path_idx][self.path_lengths[env_idx][path_idx]-1]

        input = np.hstack((obstacles, current_point, goal_point))

        return input, next_target

###### train / test utility function ######

def train_task_mpnet(model, dataloader, args, device, recon_loss_weight):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        print('[info] Epoch {}/{}'.format(epoch+1, args.epochs))
        loss_history = []
        recon_loss_history = []

        for input, targets in dataloader:
            #print(input.shape)
            #print(input[-1][-4:])
            input = torch.FloatTensor(input)
            targets = torch.FloatTensor(targets)
            input = normalize_s2d(input, 20.0)
            targets = normalize_s2d(targets, 20.0)

            input = input.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            preds, obs_rep, recon = model(input)

            task_loss = model.loss(preds, targets)
            recon_loss = model.loss(recon, obs_rep)

            total_loss = recon_loss_weight * recon_loss + (1 - recon_loss_weight) * task_loss

            total_loss.backward()
            optimizer.step()
            loss_history.append(task_loss.item())
            recon_loss_history.append(recon_loss.item())

        print('[info] Task loss {:.5f} Reconstruction loss {:.3f}'.format(np.mean(loss_history), np.mean(recon_loss_history)))

    #EXPERIMENTAL CASE
    if args.unperturbed and args.train_scheme=='task_aware':
        enc_path = os.path.join(MPNET_ROOT_DIR, 'float_models', '{}.pt'.format('mpnet_encoder_task_aware'))
        torch.save(model.encoder.state_dict(), enc_path)
        print('[info] encoder model saved')
    elif args.save_path is not None:
        enc_path = os.path.join(MPNET_ROOT_DIR, 'float_models', '{}_{}.pt'.format('tasknet_encoder', args.save_path))
        dec_path = os.path.join(MPNET_ROOT_DIR, 'float_models', '{}_{}.pt'.format('tasknet_decoder', args.save_path))
        torch.save(model.bottleneck_encoder.state_dict(), enc_path)
        torch.save(model.bottleneck_decoder.state_dict(), dec_path)
        print('[info] bottleneck model saved')
        if args.train_scheme=='end_to_end':
            custom_cae_path = os.path.join(MPNET_ROOT_DIR, 'float_models', '{}_{}.pt'.format('tasknet_cae', args.save_path))
            custom_mlp_path = os.path.join(MPNET_ROOT_DIR, 'float_models', '{}_{}.pt'.format('tasknet_mlp', args.save_path))
            torch.save(model.encoder.state_dict(), custom_cae_path)
            torch.save(model.mlp.state_dict(), custom_mlp_path)
            print('[info] custom mpnet model saved')

def test_task_mpnet(model, dataloader, args, device):
    #TODO
    # generate reconstruceted representation of obstacles
    model.eval()
    loss_history = []
    recon_loss_history = []
    with torch.no_grad():
        for input, targets in dataloader:
            input = torch.FloatTensor(input)
            targets = torch.FloatTensor(targets)
            input = normalize_s2d(input, 20.0)
            targets = normalize_s2d(targets, 20.0)

            input = input.to(device)
            targets = targets.to(device)

            preds, obs_rep, recon = model(input)
            task_loss = model.loss(preds, targets)
            recon_loss = model.loss(recon, obs_rep)

            loss_history.append(task_loss.item())
            recon_loss_history.append(recon_loss.item())

        print('[info] Task loss {:.5f} Reconstruction loss {:.3f}'.format(np.mean(loss_history), np.mean(recon_loss_history)))
        print(np.mean(loss_history))
        print(np.mean(recon_loss_history))

    result_dict = {
        'data_type': 'motion_plan_{}_{}'.format(args.env_type, 'test'),
        'net_type': 'mpnet',
        'train_scheme': args.train_scheme,
        'z_dim': args.latent_dim,
        'reconstruction_loss': np.mean(recon_loss_history),
        'task_loss': np.mean(loss_history),
    }

def plan_task_mpnet(model, dataset, IsInCollision, args, unperturbed_path):
    planned_paths = []
    feasibility_stats = []
    for i in range(dataset.paths.shape[0]):
        print('[info] start planning on env {}'.format(i))
        planned_path = []
        feasibility = []
        for j in range(dataset.paths.shape[1]):
            start_end_tuple = (torch.FloatTensor(dataset.paths[i][j][0]),
                                torch.FloatTensor(dataset.paths[i][j][dataset.path_lengths[i][j]-1]))

            fp, path, path_xy = run_MPNET(start_end_tuple, model, torch.FloatTensor(dataset.obc[i]), torch.FloatTensor(dataset.obs[i]), IsInCollision, normalize_func_s2d, unnormalize_func_s2d)
            #print('debug: {} / {}'.format(fp, path))
            planned_path.append(path)
            feasibility.append(fp)
        planned_paths.append(list(planned_path))
        feasibility_stats.append(list(feasibility))

    acc = np.count_nonzero(feasibility_stats) / np.array(feasibility_stats).size
    print('[info] {:.2f} % of all paths is feasible'.format(acc*100))
    if unperturbed_path is not None:
        path_score = path_length_score(feasibility_stats, planned_paths, unperturbed_path)
        print('[info] Path score {:.4f} '.format(path_score))
    else:
        path_score = None

    #SAVING RESULTS
    if args.unperturbed:
        if args.mpnet_path is None:
            train_scheme = 'unperturbed'
        else:
            #EXPERIMENTAL CASE: TRAIN CAE WITH TASK_AWARE
            custom_net = 'cae' if args.mpnet_path[0] is not None else ''
            custom_net += 'mlp' if len(args.mpnet_path) > 1 and args.mpnet_path[1] is not None else ''
            train_scheme = 'unperturbed_custom_{}_{}'.format(custom_net, args.train_scheme)
    else:
        train_scheme = args.train_scheme

    result_dict = {
        'data_type': 'motion_plan_{}_{}'.format(args.env_type, 'seen' if args.start_env+args.num_env <= 100 else 'unseen'),
        'net_type': 'mpnet',
        'train_scheme': train_scheme,
        'z_dim': None if args.unperturbed else args.latent_dim,
        'accuracy': acc,
        'task_loss': path_score,
    }

    #write unperturbed_path if necessary
    unperturbed_path_pkl = os.path.join(MPNET_ROOT_DIR, 'original_mpnet_path_{}.pkl'.format('seen' if args.start_env+args.num_env <= 100 else 'unseen'))
    if args.unperturbed and not os.path.exists(unperturbed_path_pkl):
        write_pkl(unperturbed_path_pkl, dict(feasibility=feasibility_stats, path=planned_paths))
    elif 'custom' in train_scheme:
        #EXPERIMENTAL
        custom_mpnet_pkl = os.path.join(MPNET_ROOT_DIR, '{}_path_{}.pkl'.format(train_scheme, 'seen' if args.start_env+args.num_env <= 100 else 'unseen'))
        write_pkl(custom_mpnet_pkl, dict(feasibility=feasibility_stats, path=planned_paths))
    elif not args.unperturbed:
        idx = 1
        planned_paths_pkl = os.path.join(MPNET_ROOT_DIR, 'task_mpnet_path_{}_{}_{}_{}.pkl'.format(args.train_scheme, args.latent_dim, 'seen' if args.start_env+args.num_env <= 100 else 'unseen', idx))
        while os.path.exists(planned_paths_pkl):
            idx += 1
            planned_paths_pkl = planned_paths_pkl.replace(planned_paths_pkl.split('_')[-1], str(idx)+'.pkl')
        write_pkl(planned_paths_pkl, dict(feasibility=feasibility_stats, path=planned_paths))

def extract_run_MPNET_args(args):
    start_end_tuple, model, obc, obs, IsInCollision, normalize_func, unnormalize_func = args
    return run_MPNET(start_end_tuple, model, torch.FloatTensor(obc), torch.FloatTensor(obs), IsInCollision, normalize_func, unnormalize_func)

def path_length(path):
    return np.sum([np.linalg.norm(path[i]-path[i+1]) for i in range(len(path)-1)])

def path_length_score(feasibility, planned_paths, unperturbed_path):
    feasibility_up = unperturbed_path['feasibility']
    planned_path_up = unperturbed_path['path']
    #assert len(feasibility) == len(feasibility_up)
    scores = []
    for i in range(len(feasibility)):
        for j in range(len(feasibility[0])):
            if feasibility[i][j] == 1 and feasibility_up[i][j] == 1:
                #print('[debug] {:.2f} / {:.2f}'.format(p_len_up, p_len_pn))
                planned_path_len = path_length(planned_paths[i][j])
                unperturbed_path_len = path_length(planned_path_up[i][j])
                if planned_path_len == 0:
                    if unperturbed_path_len == 0:
                        scores.append(1.0)
                    else:
                        print('[warn] unperturbed path length is not zero inspite of zero path length on env {} path {}'.format(i,j))
                else:
                    scores.append(unperturbed_path_len / planned_path_len)
    #print('[debug] path length score {}'.format(scores))
    return np.mean(scores)

normalize_s2d = utility_s2d.normalize
unnormalize_s2d = utility_s2d.unnormalize

def normalize_func_s2d(x):
    return normalize_s2d(x, 20.0)

def unnormalize_func_s2d(x):
    return unnormalize_s2d(x, 20.0)

def write_pkl(fname = None, input_dict = None):
    with open(fname, 'wb') as f:
        pickle.dump(input_dict, f)

def load_pkl(fname = None):
    with open(fname, 'rb') as f:
        out_dict = pickle.load(f)
    return out_dict

####### MAIN ########

def main(args):
    #instantiate mpnet
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('[info] cuda.set_device')
    else:
        device = torch.device('cpu')

    # setup evaluation function
    if args.env_type == 's2d':
        IsInCollision = plan_s2d.IsInCollision
        load_test_dataset = data_loader_2d.load_test_dataset
        load_raw_dataset = data_loader_2d.load_raw_dataset
        CAE = CAE_2d
        MLP = model.MLP
        data_dir = os.path.join(MPNET_ROOT_DIR, 'data', 'simple', 's2d/')
        mlp_path = os.path.join(MPNET_ROOT_DIR, 'hybrid_res', 'global', 's2d', 'offlinebatch', 'mlp_100_4000_PReLU_ae_dd150.pkl')
        cae_path = os.path.join(MPNET_ROOT_DIR, 'hybrid_res', 'global', 's2d', 'offlinebatch', 'cae_encoder.pkl')
        unperturbed_path_pkl = os.path.join(MPNET_ROOT_DIR, 'original_mpnet_path_{}.pkl'.format('seen' if args.start_env+args.num_env <= 100 else 'unseen'))
    #NOT IMPLEMENTED
    elif args.env_type == 'r3d':
        raise Exception('env type r3d is not implemented!')
        IsInCollision = plan_r3d.IsInCollision
        load_test_dataset = data_loader_r3d.load_test_dataset
        normalize = utility_r3d.normalize
        unnormalize = utility_r3d.unnormalize
        CAE = CAE_r3d
        MLP = model.MLP
        data_dir = None#TODO
        mlp_path = None
        cae_path = None

    #instantiate model
    latent_dim = None if args.unperturbed else args.latent_dim
    if args.mode in ['train', 'test']:
        task_mpnet = TaskMPNet(total_input_size=2804, AE_input_size=2800, mlp_input_size=28+4, output_size=2, AEtype='deep', n_tasks=1, n_memories=256, memory_strength=0.5, grad_step=1, CAE=CAE, MLP=MLP, latent_dim=latent_dim, get_z=True)
        if args.bottleneck_path is not None and len(args.bottleneck_path) > 1:
            task_mpnet.bottleneck_encoder.load_state_dict(torch.load(args.bottleneck_path[0]))
            task_mpnet.bottleneck_decoder.load_state_dict(torch.load(args.bottleneck_path[1]))
            print('[info] pretrain bottleneck loaded')
        #EXPERIMENTAL CASE: TRAIN ORIGINAL ENCODER WITH TASK_AWARE MANNER
        if args.unperturbed and args.train_scheme=='task_aware':
            task_mpnet.mlp.load_state_dict(torch.load(mlp_path))
            task_mpnet.freeze_mpnet_params('mlp')
        elif args.train_scheme != 'end_to_end':
            task_mpnet.mlp.load_state_dict(torch.load(mlp_path))
            task_mpnet.encoder.load_state_dict(torch.load(cae_path))
            print('[info] pretrain weights loaded')
            task_mpnet.freeze_mpnet_params()
        else:
            assert args.train_scheme == 'end_to_end'
            if args.mpnet_path is not None:
                task_mpnet.encoder.load_state_dict(torch.load(args.mpnet_path[0]))
                task_mpnet.mlp.load_state_dict(torch.load(args.mpnet_path[1]))
                print('[info] pretrain custom mpnet loaded')
    else:
        assert args.mode=='plan'
        task_mpnet = TaskMPNet(total_input_size=2804, AE_input_size=2800, mlp_input_size=28+4, output_size=2, AEtype='deep', n_tasks=1, n_memories=256, memory_strength=0.5, grad_step=1, CAE=CAE, MLP=MLP, latent_dim=latent_dim, get_z=False)
        if args.mpnet_path is not None:
            if args.train_scheme != 'end_to_end':
                print('[warn] trying to load custom mpnet weights in case of {}'.format(args.train_scheme))
            if len(args.mpnet_path) > 0 and args.mpnet_path[0] is not None:
                task_mpnet.encoder.load_state_dict(torch.load(args.mpnet_path[0]))
                print('[info] custom encoder weights loaded')
            else:
                task_mpnet.encoder.load_state_dict(torch.load(cae_path))
                print('[info] default encoder weights loaded')
            if len(args.mpnet_path) > 1 and args.mpnet_path[1] is not None:
                task_mpnet.mlp.load_state_dict(torch.load(args.mpnet_path[1]))
                print('[info] custom mlp weights loaded')
            else:
                task_mpnet.mlp.load_state_dict(torch.load(mlp_path))
                print('[info] default mlp weights loaded')
        else:
            task_mpnet.mlp.load_state_dict(torch.load(mlp_path))
            task_mpnet.encoder.load_state_dict(torch.load(cae_path))
            print('[info] default mpnet weights loaded')
        if not args.unperturbed:
            task_mpnet.bottleneck_encoder.load_state_dict(torch.load(args.bottleneck_path[0]))
            task_mpnet.bottleneck_decoder.load_state_dict(torch.load(args.bottleneck_path[1]))
            print('[info] bottleneck weights loaded')

    if torch.cuda.is_available():
        print('[info] transfer models to gpu')
        task_mpnet.to(device)

    #load_dataset
    obc, obs, paths, path_lengths = load_test_dataset(N=args.num_env, NP=args.num_path, s=args.start_env, sp=args.start_path, folder=data_dir)
    if args.env_type == 's2d':
        dataset = S2DDataset(obc, obs, paths, path_lengths)
        dataloader = torch.utils.data.DataLoader(dataset, args.batch_size, shuffle=True if args.mode=='train' else False)
    #run tasknet
    if args.mode=='train':
        if args.train_scheme in ['task_aware', 'end_to_end']:
            recon_loss_weight = 0
        elif args.train_scheme=='task_agnostic':
            recon_loss_weight = 1
        else:
            recon_loss_weight = 0.1
        train_task_mpnet(task_mpnet, dataloader, args, device, recon_loss_weight)
    elif args.mode=='test':
        test_task_mpnet(task_mpnet, dataloader, args, device)
    else:
        assert args.mode=='plan'
        if os.path.exists(unperturbed_path_pkl):
            unperturbed_path = load_pkl(unperturbed_path_pkl)
        else:
            unperturbed_path = None
        plan_task_mpnet(task_mpnet, dataset, IsInCollision, args, unperturbed_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_env', type=int, default=1)
    parser.add_argument('--num_path', type=int, default=1)
    parser.add_argument('--start_env', type=int, default=0)
    parser.add_argument('--start_path', type=int, default=0)
    parser.add_argument('--env_type', type=str, default='s2d', help='s2d for simple 2d, c2d for complex 2d')
    parser.add_argument('--train_scheme', type=str, choices=['task_aware', 'task_agnostic', 'task_aware_weighted', 'end_to_end'], help='train scheme with which MPNET is trained')
    parser.add_argument('--latent_dim', type=int)
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'plan'])
    parser.add_argument('--batch_size', type=int, default=8192)
    parser.add_argument('--lr', type=float, default=0.004)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--bottleneck_path', nargs='+', help='path to bottleneck encoder/decoder weights. (ENCODER, DECODER)')
    parser.add_argument('--mpnet_path', nargs='+', help='path to custom MPNet weights. (CAE, MLP)')
    parser.add_argument('--unperturbed', action='store_true', help='mode must be "plan" if specified')

    args = parser.parse_args()
    print('start')
    main(args)
    print('done')
