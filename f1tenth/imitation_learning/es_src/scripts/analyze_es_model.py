####  LOAD MODEL  ####
import os
import json
import numpy as np
from argparse import Namespace

run = 28
epoch = 120
budget = 19200
worker_num = 1
objective_num = 2
map_name = 'General1'
wpt_name = 'traj_race_cl.csv'
render_sim = True
save_log = False
ego_idx = -1

# store_dir = 'es_model'
store_dir = 'data'
data_module = os.path.abspath(f'../{store_dir}')
if store_dir=='es_model':
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
####  LOAD MODEL  ####


####  RUN LATTICE PLANNER  ####
from es.planner.lattice_planner import LatticePlanner
import gym
from es.utils.DataProcessor import RolloutDataLogger
from es.utils.utils import *
from es.utils.visualize import LatticePlannerRender
from es.worker import calculate_objectives

module_path = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'configs')
# fix the config path if needed
mapconfig_path = os.path.join(module_path, map_name)
config.map_path = os.path.join(mapconfig_path, map_name + '_map')
config.wpt_path = os.path.join(mapconfig_path, wpt_name)
config.tracker_config_path = os.path.join(mapconfig_path, 'pure_pursuit_config.yaml')
# load planner
ego_planner = LatticePlanner(config)
opp_planner = LatticePlanner(config)
ego_params = all_seeds[ego_id].copy()
# ego_params[1:] = 1.0
# ego_params[0] = 0.9
ego_planner.set_parameters(ego_params)
seed_dict = {}
for key, value in zip(config.params_dict.values(), ego_params):
    seed_dict[key] = np.round(value, 4)
print(f'set seed {seed_dict}')
print(f'score is {all_scores[ego_id]}')
print(f'overtake is {all_overtake[ego_id]}')
print(f'crash is {all_crash[ego_id]}')
env = gym.make('f110_gym:f110-v0', map=config.map_path, map_ext='.png', num_agents=2)
render = LatticePlannerRender(ego_planner)
logger = RolloutDataLogger(rollout_states=config.rollout_states, log_location=batch_data_location)
env.add_render_callback(render.render_callback)

batch_score = []
batch_crash = 0
batch_overtake = 0


# o_idx also serve as the idx of different scenarios
for o_idx in range(2, opp_num):
    ### init for a rollout ###
    print(f'opponent v is {opp_weights[o_idx][0]}')
    logger.clear_buffer()
    opp_planner.set_parameters(opp_weights[o_idx])
    agent_poses = np.vstack((ego_start_pose[o_idx], opp_start_pose[o_idx]))
    obs, _, done, _ = env.reset(agent_poses)
    opp_planner.init_rollout()
    ego_planner.init_rollout()
    laptime = 0
    ego_s = 0.0
    opp_s = 0.0
    _, last_dist = cal_single_ittc(pose_1=ego_start_pose[o_idx].flatten(), pose_2=opp_start_pose[o_idx].flatten(),
                                   dt=0.01 * config.tracker_steps)
    step_states = {
        'rollout_obs': obs
    }
    logger.update_buffer(step_states)
    overtake = False
    crash = False
    rollout_step = 0

    ### init for a rollout ###
    if render_sim:
        env.render('human')
    while not done and laptime < config.time_per_run:
        rollout_step += 1
        ######### lattice plan ########
        oppo_pose = obsDict2oppoArray(obs, 0)
        ego_best_traj = ego_planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], oppo_pose,
                                         obs['linear_vels_x'][0])

        oppo_pose = obsDict2oppoArray(obs, 1)
        opp_best_traj = opp_planner.plan(obs['poses_x'][1], obs['poses_y'][1], obs['poses_theta'][1], oppo_pose,
                                         obs['linear_vels_x'][1])

        tracker_count = 0
        while not done and tracker_count < config.tracker_steps:
            ego_steer, ego_speed = ego_planner.tracker.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0],
                                                            obs['linear_vels_x'][0], ego_best_traj)
            opp_steer, opp_speed = opp_planner.tracker.plan(obs['poses_x'][1], obs['poses_y'][1], obs['poses_theta'][1],
                                                            obs['linear_vels_x'][1], opp_best_traj)
            action = np.array([[ego_steer, ego_speed], [opp_steer, opp_speed]])
            obs, timestep, done, _ = env.step(action)
            if np.any(obs['collisions']):
                done = True
            if tracker_count != 0:
                step_states = {
                    'ego_control_error': ego_planner.tracker.nearest_dist
                }
                logger.update_buffer(step_states)
            laptime += timestep
            tracker_count += 1
            if render_sim:
                env.render('human')

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

        if ego_s > opp_s and rollout_step < 2:
            opp_s = opp_s + ego_planner.s_max
            opp_planner.last_s = opp_s

        # ittc with oppo
        if ego_s < opp_s:
            ittc, last_dist = cal_single_ittc(
                pose_1=np.array((obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0])),
                pose_2=np.array((obs['poses_x'][1], obs['poses_y'][1], obs['poses_theta'][1])),
                last_dist=last_dist, dt=0.01 * config.tracker_steps
            )
            if ittc <= 0.0:
                ittc = 0.0
        else:
            ittc = np.nan

        # update buffer
        step_states = {
            'rollout_obs': obs,
            'ego_best_traj_idx': ego_planner.best_traj_idx,
            'opp_best_traj_idx': opp_planner.best_traj_idx,
            'ego_prev_traj': ego_planner.prev_traj_local,
            'ego_ittc': ittc,
            'rollout_ego_s': ego_s,
            'rollout_opp_s': opp_s,
            'abs_ittc': abs_ittc
        }
        logger.update_buffer(step_states)

        # if render_sim:
        #     print(ego_planner.step_all_cost)

    # calculate objective
    if ego_s > opp_s:
        overtake = True
        batch_overtake += 1
    if obs['collisions'][0] and not overtake:
        crash = True
        batch_crash += 1
    logger.remove_nan('ego_ittc')
    rollout_objectives = ego_planner.cal_objectives(logger.rollout_buffer, laptime, overtake, crash)
    if rollout_objectives[0] == 0 and rollout_objectives[1] == 0:
        print(f'invalid rollout, crash at first or overflow')
        continue
    print(
        f'scene_id:{1}, oppo_id:{o_idx}, laptime:{np.round(laptime, 2)}, objectives:{np.round(rollout_objectives, 2)}',
        f'overtake is {overtake}, crash is {crash}')
    # save rollout data
    features = {
        'ego_weights': all_seeds[ego_id],
        'opp_weights': opp_weights[o_idx],
        'objectives': rollout_objectives
    }
    if save_log:
        logger.save_rollout_data(o_idx=o_idx, s_idx=0, **features)
    batch_score.append(rollout_objectives)
if save_log:
    np.savez_compressed(os.path.join(batch_data_location, 'all_rollout_scores.npz'), rollout_scores=batch_score)

n = len(batch_score)
mean_obj = np.mean(batch_score, axis=0)
mean_obj[0] += 5.0 * (n - batch_overtake) / n
mean_obj[1] += 10.0 * batch_crash / n
print(f'batch score: {mean_obj}, batch_overtake:{batch_overtake}, batch_crash{batch_crash}')
####  RUN LATTICE PLANNER  ####
