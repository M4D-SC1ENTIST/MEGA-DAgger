import numpy as np
from es.utils.utils import cal_ittc
"""
rollout_data:
    np.savez_compressed(npz_file, ego_x=ego_state['pose_x'], ego_y=ego_state['pose_y'],
                        ego_theta=ego_state['pose_theta'], ego_v=ego_state['v'],
                        opp_x=opp_state['pose_x'], opp_y=opp_state['pose_y'],
                        opp_theta=opp_state['pose_theta'], opp_v=opp_state['v'],
                        ego_weights=ego_planner.cost_weights, opp_weights=opp_planner.cost_weights, objective=rollout_objectives)
"""
import os
from es.utils.visualize import draw_pts, value2color
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import fire
from es.planner.lattice_planner import *
from es.utils.utils import *
from es.utils.visualize import draw_traj_with_cost, draw_traj_with_state
from es.utils.DataProcessor import obsDict2carStateSeq
from es.worker import calculate_objectives
from datetime import datetime


def load_rollout_data_color():
    # sns.color_palette("rocket_r", as_cmap=True)
    v_min = -1.0
    v_max = 8.0
    v_norm = matplotlib.colors.Normalize(vmin=v_min, vmax=v_max, clip=True)
    npz_file = os.path.join(os.path.abspath('.'), 'test_rollout.npz')
    rollout_data = np.load(npz_file, mmap_mode='r')
    states = {}
    for key in ('ego_x', 'ego_y', 'ego_v', 'opp_x', 'opp_y', 'opp_v', 'ref_v'):
        states[key] = rollout_data[key]
    states_df = pd.DataFrame(states)
    return states_df, v_norm


def analyze_rollout_v():
    states_df, v_norm = load_rollout_data()
    traj_fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    #### traj
    sns.scatterplot(ax=axes[0], data=states_df, x='ego_x', y='ego_y', label='ego', size=0.05)
    sns.scatterplot(ax=axes[0], data=states_df, x='opp_x', y='opp_y', label='opp', size=0.05)
    axes[0].axis('equal')

    #### velocity
    axes[1].plot(states_df['ego_v'], '-o', label='ego_v', markersize=0.2)
    axes[1].plot(states_df['opp_v'], '-o', label='opp_v', markersize=0.2)
    axes[1].plot(states_df['ref_v'], '-o', label='ref_v', markersize=0.2)
    plt.legend()
    plt.show()


def load_rollout_data(run=0, s=0, o=0, epoch=0, worker=0):
    npz_file = os.path.join(os.path.abspath('..'), 'runs', f'{run}', 'batch_data', f'worker_{worker}', f'epoch_{epoch}', f's_{s}_o_{o}.npz')
    rollout_data = np.load(npz_file, allow_pickle=True, mmap_mode='r')
    states = {}
    for key in ('ego_x', 'ego_y', 'ego_v', 'ego_theta', 'opp_x', 'opp_y', 'opp_v', 'opp_theta', 'ego_prev_traj', 'ego_prev_opp', 'ego_ittc'):
        states[key] = rollout_data[key]

    feature = {}
    for key in ('ego_weights', 'opp_weights', 'objective'):
        feature[key] = rollout_data[key]
    return states, feature


def get_poses_for_step(states, step):
    ego_pose = np.array([states['ego_x'][step], states['ego_y'][step], states['ego_theta'][step]])
    opp_pose = np.array([states['opp_x'][step], states['opp_y'][step], states['opp_theta'][step]])
    return np.vstack((ego_pose, opp_pose))


def replay():
    ## state setting
    seed = 6300
    rng = np.random.default_rng(seed)
    states, feature = load_rollout_data()
    n = len(states['ego_x'])
    print(f'length of rollout {n}')
    map_name = 'General1'
    lattice_planner_conf_path = \
        os.path.join(os.path.abspath('..'), 'configs', map_name, 'lattice_planner_config.yaml')
    with open(lattice_planner_conf_path) as file:
        lp_conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    lp_conf = Namespace(**lp_conf_dict)
    ego_planner = LatticePlanner(lp_conf)
    opp_planner = LatticePlanner(lp_conf)
    waypoints = ego_planner.waypoints
    waypoints_xytheta = np.hstack((waypoints[:, :2], waypoints[:, 3].reshape(-1, 1)))

    ##
    # ego_planner.set_parameters(feature['ego_weights'])
    # opp_planner.set_parameters(feature['opp_weights'])

    ##
    step = 25
    poses = get_poses_for_step(states, step)
    ego_planner.prev_traj_local = states['ego_prev_traj'][step]
    ego_planner.prev_opp_pose = states['ego_prev_opp'][step]
    draw_traj_with_cost(waypoints_xytheta, ego_planner, poses, states['ego_v'][step], states['opp_v'][step],
                        clip_mapcost=True, traj_id=-1)


def analyze_rollout_feature(run=26, s=0, o=2, epoch=0, worker=0, feature_name='ego_ittc'):
    states, feature = load_rollout_data(run, s, o, epoch, worker)
    seed = 6300
    n = len(states['ego_x'])
    print(f'length of rollout {n}')
    map_name = 'General1'
    lattice_planner_conf_path = \
        os.path.join(os.path.abspath('..'), 'configs', map_name, 'lattice_planner_config.yaml')
    with open(lattice_planner_conf_path) as file:
        lp_conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    lp_conf = Namespace(**lp_conf_dict)
    ego_planner = LatticePlanner(lp_conf)
    opp_planner = LatticePlanner(lp_conf)
    waypoints = ego_planner.waypoints
    waypoints_xytheta = np.hstack((waypoints[:, :2], waypoints[:, 3].reshape(-1, 1)))
    # import ipdb;ipdb.set_trace()
    # ego_planner.set_parameters(feature['ego_weights'])
    # opp_planner.set_parameters(feature['opp_weights'])

    step = n // 2
    poses = get_poses_for_step(states, step)
    ego_planner.prev_traj_local = states['ego_prev_traj'][step]
    ego_planner.prev_opp_pose = states['ego_prev_opp'][step]
    ego_traj = np.vstack((states['ego_x'], states['ego_y']))
    opp_traj = np.vstack((states['opp_x'], states['opp_y']))
    draw_traj_with_state(waypoints_xytheta, ego_planner, poses, ego_traj, opp_traj, states[feature_name], feature_name)


if __name__ == '__main__':
    fire.Fire(
        {'velocity': analyze_rollout_v,
         'feature': analyze_rollout_feature}
    )
