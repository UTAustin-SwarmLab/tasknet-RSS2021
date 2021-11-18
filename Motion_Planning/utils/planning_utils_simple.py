"""
    key utils for path generation of motion planner MPNET
    MAIN FILE TO READ
"""

import sys,os
import numpy as np
DEFAULT_STEP = 2.
MAX_NEURAL_REPLAN = 11
SMALL_STEP = 0.1

import torch

sys.path.append(os.environ['MPNET_LIB'])
from utility import *
from plan_general import *

def get_path_xy(path):
    path_x = [k[0] for k in path]
    path_y = [k[1] for k in path]
    path_xy = [path_x, path_y]

    return path_xy

"""
    convert a path as a torch tensor to a numpy array
"""

def get_numpy_vector(path):

    if type(path[0]) is not np.ndarray:
        # it is torch tensor, convert to numpy
        path = [p.numpy() for p in path]
    path = np.array(path)

    return path

"""
    run MPnet open-loop from start to end, need to add collision checking later
"""

def simple_feed_forward_planner(mpNet, start, goal, obc, obs, IsInCollision, normalize, unnormalize, MAX_LENGTH, step_sz=None):
    # plan a mini path from start to goal
    # obs: tensor
    itr=0
    pA=[]
    pA.append(start)
    target_reached=0
    new_path = []

    while target_reached==0 and itr<MAX_LENGTH:
        itr=itr+1  # prevent the path from being too long

        ip1=torch.cat((obs,start,goal)).unsqueeze(0)

        ip1=normalize(ip1)

        ip1=to_var(ip1)

        start=mpNet(ip1).squeeze(0)

        start=start.data.cpu()

        start = unnormalize(start)

        pA.append(start)

        target_reached=steerTo(start, goal, obc, IsInCollision, step_sz=step_sz)

    if target_reached==0:
        return 0
    else:
        for p1 in range(len(pA)):
            new_path.append(pA[p1])
        new_path.append(goal)
    return new_path

"""
    main wrapper to run the Motion planner
    - it re-tries if collisions are found and uses lvc (lazy vertex contraction)
    to remove redundant nodes
    - this is the full algorithm used in the paper
"""

def run_MPNET(start_end_tuple, mpNet, obstacle_cloud, obstacle_encoding, IsInCollision, normalize_func, unnormalize_func, step_sz=DEFAULT_STEP, time_flag=False):

    start_coord = start_end_tuple[0]
    end_coord = start_end_tuple[1]

    fp = 0 # indicator for feasibility
    vp = 1

    path = [start_coord, end_coord]

    # try this 11 times
    for t in range(MAX_NEURAL_REPLAN):
    # adaptive step size on replanning attempts
        if (t == 2):
            step_sz = 1.2
        elif (t == 3):
            step_sz = 0.5
        elif (t > 3):
            step_sz = 0.1

        path = neural_replan(mpNet, path, obstacle_cloud, obstacle_encoding, IsInCollision, normalize_func, unnormalize_func, t==0, step_sz=step_sz, time_flag=time_flag)

        path = lvc(path, obstacle_cloud, IsInCollision, step_sz=step_sz)

        if feasibility_check(path, obstacle_cloud, IsInCollision, step_sz=0.01):
            fp = 1
            #print('feasible, ok!')
            break

    if type(path[0]) is not np.ndarray:
        # it is torch tensor, convert to numpy
        path = [p.numpy() for p in path]
    path = np.array(path)

    # break path in xy coordinates
    path_xy = get_path_xy(path)

    return fp, path, path_xy

"""
    call MPNet once without re-tries, this is not in the full paper but just our
    comparison
"""

def run_open_loop_path(start_end_tuple, mpNet, obstacle_cloud, obstacle_encoding, IsInCollision, normalize_func, unnormalize_func, MAX_LENGTH=80, step_sz = 0.01):

    start_coord = start_end_tuple[0]
    end_coord = start_end_tuple[1]

    mini_path, time_d = neural_replanner(mpNet, start_coord, end_coord, obstacle_cloud, obstacle_encoding, IsInCollision, normalize_func, unnormalize_func, MAX_LENGTH)

    if mini_path:
        open_loop_path_tensor = removeCollision(mini_path, obstacle_cloud, IsInCollision)
    else:
        open_loop_path_tensor = [start_coord, end_coord]

    feasible_open_loop_path = feasibility_check(open_loop_path_tensor, obstacle_cloud, IsInCollision, step_sz=step_sz)

    open_loop_path = get_numpy_vector(open_loop_path_tensor)

    # break path in xy coordinates
    open_loop_path_xy = get_path_xy(open_loop_path)
    return feasible_open_loop_path, open_loop_path, open_loop_path_xy

"""
    this is a wrapper of the stochastic open loop planner
    run for N_TRIALS to get a distribution of trajectories
"""

def run_several_trials_stochastic_planner(start_end_tuple, mpNet, obstacle_cloud, obstacle_encoding, IsInCollision, normalize_func, unnormalize_func, MAX_LENGTH=80, step_sz = 0.01, NUM_TRIALS=10):

    start_coord = start_end_tuple[0]
    end_coord = start_end_tuple[1]

    stochastic_path_list = []

    feasible_vec = []

    for i in range(NUM_TRIALS):
        stochastic_path_tensor = simple_feed_forward_planner(mpNet, start_coord, end_coord, obstacle_cloud, obstacle_encoding, IsInCollision, normalize_func, unnormalize_func, MAX_LENGTH, step_sz=SMALL_STEP)

        stochastic_path = get_numpy_vector(stochastic_path_tensor)

        feasible_path = feasibility_check(stochastic_path_tensor, obstacle_cloud, IsInCollision, step_sz=0.01)

        stochastic_path_xy = get_path_xy(stochastic_path)

        stochastic_path_list.append(stochastic_path_xy)
        feasible_vec.append(feasible_path)

    feasible_mean = np.mean(feasible_vec)

    return stochastic_path_list, feasible_vec, feasible_mean


"""
    run several stochastic trials of the MPNET planner
"""

def run_several_trials_MPNET_planner(start_end_tuple, mpNet, obstacle_cloud, obstacle_encoding, IsInCollision, normalize_func, unnormalize_func, step_sz = 0.01, NUM_TRIALS=10):

    start_coord = start_end_tuple[0]
    end_coord = start_end_tuple[1]

    MPNET_path_list = []
    feasible_vec = []

    for i in range(NUM_TRIALS):
        feasible_path, MPNET_path, MPNET_path_xy = run_MPNET(start_end_tuple, mpNet, obstacle_cloud, obstacle_encoding, IsInCollision, normalize_func, unnormalize_func, step_sz=DEFAULT_STEP, time_flag=False)

        MPNET_path_list.append(MPNET_path_xy)
        feasible_vec.append(feasible_path)

    feasible_mean = np.mean(feasible_vec)

    return MPNET_path_list, feasible_vec, feasible_mean
