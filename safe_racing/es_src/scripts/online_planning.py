#### LOAD MODEL ####
import os
import json
import numpy as np
from argparse import Namespace

###### Set Parameters #######
# ego model
run = 26
epoch = 120
budget = 19200
worker_num = 1
objective_num = 2
map_name = 'map0'
wpt_name = 'traj_race_cl.csv'
render_sim = True
save_log = False
ego_idx = -1
store_dir = 'es_model'
# store_dir = 'es_model'

# seed
seed = 6300
rng = np.random.default_rng(seed)
xy_noise_scale = 0.0
theta_noise_scale = 0.0

# online parameter
evaluate_rollout_num = 100
online_switch_horizon_step = 100  # step
online_rollout_horizon_t = 16  # laptime

###### Set Parameters #######

### Set scenario function ###

def random_position(waypoints_xytheta, sampled_number=1, rng=None, xy_noise=0.0, theta_noise=0.0):
    ego_idx = rng.choice(np.arange(0, len(waypoints_xytheta)), 1)[0]
    print(f'ego_idx is {ego_idx}')
    for i in range(sampled_number):
        starting_idx = (ego_idx - i * 10) % len(waypoints_xytheta)
        x, y, theta = waypoints_xytheta[starting_idx][0], waypoints_xytheta[starting_idx][1], \
                      waypoints_xytheta[starting_idx][2]
        x = x + rng.random(size=1)[0] * xy_noise
        y = y + rng.random(size=1)[0] * xy_noise
        theta = (zero_2_2pi(theta) + 0.5 * np.pi) + rng.random(size=1)[0] * theta_noise
        if i == 0:
            res = np.array([[x, y, theta]])  # (1, 3)
        else:
            res = np.vstack((res, np.array([[x, y, theta]])))
    return res, ego_idx


### Set scenario function ###


#### Load Prototypes ####
data_module = os.path.abspath(f'../{store_dir}')
if store_dir == 'es_model':
    score_file = f'{run}/default_CMA_budget_{budget}epoch{epoch}_score.npz'
    model_file = f'{run}/default_CMA_budget_{budget}epoch{epoch}_optim.pkl'
    oppo_file = f'{run}/default_CMA_budget_{budget}_opp_weights.npz'
else:
    score_file = f'{run}/scores/default_CMA_budget_{budget}epoch{epoch}_score.npz'
    model_file = f'{run}/optims_pkl/default_CMA_budget_{budget}epoch{epoch}_optim.pkl'
    oppo_file = f'{run}/scores/default_CMA_budget_{budget}_opp_weights.npz'
config_file = f'{run}/config.json'
scenario_file = f'{run}/rollout_scenarios.npz'
pareto_file = f'{run}/near_pareto_idx.npz'
batch_file_dir = f'{run}/batch_data/{ego_idx}'
os.makedirs(os.path.join(data_module, batch_file_dir), exist_ok=True)
batch_data_location = os.path.join(data_module, batch_file_dir)

config_data = json.load(open(os.path.join(data_module, config_file)))
config = Namespace(**config_data)
# TODO: tmp
# config.ittc_thres = 1
opp_weights = np.load(os.path.join(data_module, oppo_file), mmap_mode='r')['opp_weights']
opp_num = len(opp_weights)
score_data = np.load(os.path.join(data_module, score_file), mmap_mode='r', allow_pickle=True)
all_scores = score_data['scores']  # (num of seeds, num of scores)
all_seeds = score_data['params']  # (num of seeds, num of params)
all_overtake = score_data['overtake']
all_crash = score_data['crash']
scenario_data = np.load(os.path.join(data_module, scenario_file), mmap_mode='r', allow_pickle=True)
ego_start_pose = scenario_data['ego_start_pose']
opp_start_pose = scenario_data['opp_start_pose']
pareto_idx = np.load(os.path.join(data_module, pareto_file), mmap_mode='r', allow_pickle=True)['near_idx']

ego_id = pareto_idx[ego_idx]
print(len(pareto_idx))
#### Load Prototypes ####


####  Set PLANNER  ####
from es.planner.lattice_planner import LatticePlanner
from es.planner.lane_switcher import LaneSwitcher
import gym
from es.utils.DataProcessor import RolloutDataLogger
from es.utils.utils import *
from es.utils.visualize import LatticePlannerRender, LaneSwitcherRender
from es.worker import calculate_objectives

module_path = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'configs')
mapconfig_path = os.path.join(module_path, map_name)
config.map_path = os.path.join(mapconfig_path, map_name + '_map')
config.wpt_path = os.path.join(mapconfig_path, wpt_name)
config.tracker_config_path = os.path.join(mapconfig_path, 'pure_pursuit_config.yaml')

# ego planner
ego_planner = LatticePlanner(config)
ego_params = all_seeds[ego_id].copy()
ego_planner.set_parameters(ego_params)
waypoints = ego_planner.waypoints
waypoints_xytheta = np.hstack((waypoints[:, :2], waypoints[:, 3].reshape(-1, 1)))
seed_dict = {}
for key, value in zip(config.params_dict.values(), ego_params):
    seed_dict[key] = np.round(value, 4)
print(f'set seed {seed_dict}')
print(f'score is {all_scores[ego_id]}')
print(f'overtake is {all_overtake[ego_id]}')
print(f'crash is {all_crash[ego_id]}')

# oppo planner
# set other planner here
# opp_planner = LatticePlanner(config)
# opp_planner.set_parameters(ego_params)
import yaml
# map_name = 'General1'
lane_switcher_conf_path = \
    os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', map_name, 'lane_switcher_config.yaml')
with open(lane_switcher_conf_path) as file:
    lane_switcher_conf_dict = yaml.load(file, Loader=yaml.FullLoader)
lane_switcher_conf = Namespace(**lane_switcher_conf_dict)
opp_planner = LaneSwitcher(conf=lane_switcher_conf)
####  Set PLANNER  ##

env = gym.make('f110_gym:f110-v0', map=config.map_path, map_ext='.png', num_agents=2)
# render = LatticePlannerRender(ego_planner)
render = LaneSwitcherRender(opp_planner)
logger = RolloutDataLogger(rollout_states=config.rollout_states, log_location=batch_data_location)
env.add_render_callback(render.render_callback)

##


### Evaluation ###
win_times = 0
for i in range(evaluate_rollout_num):
    win = False
    ### set oppo planner here ###
    # opp_planner.set_parameters(opp_weights[i])
    # opp_planner.init_rollout()
    ### set oppo planner here ###

    init_poses, _ = random_position(waypoints_xytheta, 2, rng, xy_noise_scale, theta_noise_scale)
    obs, _, done, _ = env.reset(init_poses)
    ego_planner.init_rollout()
    logger.clear_buffer()

    laptime = 0
    ego_s = 0.0
    opp_s = 0.0
    s_max = ego_planner.s_max
    ego_progress = 0.0
    opp_progress = 0.0
    ego_start_s = 0.0
    opp_start_s = 0.0
    rollout_step = 0

    step_states = {
        'rollout_obs': obs
    }
    logger.update_buffer(step_states)

    if render_sim:
        env.render('human')

    while not done and laptime < online_rollout_horizon_t:
        rollout_step += 1
        online_switch_counter = 0
        """
        TODO: insert regret model here, can use information in logger
        """

        while not done and online_switch_counter < online_switch_horizon_step and laptime < online_rollout_horizon_t:
            # print(laptime)
            ######### lattice plan ########
            oppo_pose = obsDict2oppoArray(obs, 0)
            ego_best_traj = ego_planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], oppo_pose,
                                             obs['linear_vels_x'][0])

            ## oppo planner here
            oppo_pose = obsDict2oppoArray(obs, 1)
            opp_best_traj = opp_planner.plan(obs['poses_x'][1], obs['poses_y'][1], obs['poses_theta'][1], oppo_pose,
                                             obs['linear_vels_x'][1])
            ## oppo planner here

            tracker_count = 0
            while not done and tracker_count < config.tracker_steps:
                ego_steer, ego_speed = ego_planner.tracker.plan(obs['poses_x'][0], obs['poses_y'][0],
                                                                obs['poses_theta'][0],
                                                                obs['linear_vels_x'][0], ego_best_traj)
                ## oppo planner here
                opp_steer, opp_speed = opp_planner.tracker.plan(obs['poses_x'][1], obs['poses_y'][1],
                                                                obs['poses_theta'][1],
                                                                obs['linear_vels_x'][1], opp_best_traj)
                ## oppo planner here
                action = np.array([[ego_steer, ego_speed], [opp_steer, opp_speed]])
                obs, timestep, done, _ = env.step(action)
                # print(action)
                if np.any(obs['collisions']):
                    done = True
                if tracker_count != 0:
                    step_states = {
                        'ego_control_error': ego_planner.tracker.nearest_dist
                    }
                    logger.update_buffer(step_states)
                laptime += timestep
                online_switch_counter += 1
                tracker_count += 1
                if render_sim:
                    env.render('human')
            #
            ego_distance = obs['scans'][0]  # (scan_num, 1)
            ego_speed_proj = np.cos(ego_planner.angle_span) * obs['linear_vels_x'][0]  # (scan_num, 1)
            ego_speed_proj[ego_speed_proj <= 0.0] = 0.001
            raw_ittc = ego_distance / ego_speed_proj
            # abs_ittc = np.min(raw_ittc)
            if np.min(raw_ittc) > ego_planner.ittc_thres:
                abs_ittc = 0.0
            else:
                abs_ittc = np.min(raw_ittc)

            # progress
            ego_s = ego_planner.cal_s()
            opp_s = opp_planner.cal_s()
            if ego_s > opp_s and rollout_step == 1:
                opp_s = opp_s + ego_planner.s_max
                opp_planner.last_s = opp_s
            if rollout_step == 1:
                ego_start_s = ego_s
                opp_start_s = opp_s

            # update buffer
            step_states = {
                'rollout_obs': obs,
                'ego_best_traj_idx': ego_planner.best_traj_idx,
                'ego_prev_traj': ego_planner.prev_traj_local,
                'rollout_ego_s': ego_s,
                'rollout_opp_s': opp_s,
                'abs_ittc': abs_ittc
            }
            logger.update_buffer(step_states)

            # update progress
            ego_progress = ego_s - ego_start_s
            opp_progress = opp_s - opp_start_s

    if not obs['collisions'][0] and obs['collisions'][1]:
        win = True
    elif obs['collisions'][0]:
        pass
    else:
        if ego_progress > opp_progress:
            win = True

    if win:
        win_times += 1
        print(f'win at {i} rollout')

win_rate = win_times / evaluate_rollout_num
print(f'current win rate is:  {win_rate}')
