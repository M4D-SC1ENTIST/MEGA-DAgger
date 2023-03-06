import torch
import gym
import numpy as np
import argparse
import yaml

from il_utils.policies.agents.agent_mlp import AgentPolicyMLP
from il_utils.policies.experts.expert_waypoint_follower import ExpertWaypointFollower

import il_utils.utils.env_utils as env_utils

# hg-dagger import
from il_utils.dataset import Dataset
from il_utils.utils import agent_utils
from pathlib import Path

# Lane switcher expert import
import os
import json
import numpy as np
from argparse import Namespace

from es.planner.lattice_planner import LatticePlanner
from es.planner.lane_switcher import LaneSwitcher
import gym
from es.utils.DataProcessor import RolloutDataLogger
from es.utils.utils import *
from es.utils.visualize import LatticePlannerRender, LaneSwitcherRender
from es.worker import calculate_objectives

import utils

# Settings for testing the expert performance
render_sim = True
ego_oppo_collision_prob = 0.0 # Probability of collision between ego vehicle and opponent vehicle
collision_trigger_dist_thresh = 5 # Distance threshold for triggering collision
undesired_overtake_behavior_prob = 0.5 # Probability of high speed overtaking

num_iter = 100
num_ego_oppo_collision = 0
num_ego_env_collision = 0
num_overtake = 0

model_path = 'models/multi_weak_expert_conflict_check/iter_1000_model.pkl'

# seed
seed = 0
rng = np.random.default_rng(seed)
xy_noise_scale = 0.0
theta_noise_scale = 0.0


def random_position(waypoints_xytheta, sampled_number=1, rng=None, xy_noise=0.0, theta_noise=0.0):
    ego_idx = rng.choice(np.arange(0, len(waypoints_xytheta)), 1)[0]
    # ego_idx = 790
    # print(f'ego_idx is {ego_idx}')
    for i in range(sampled_number):
        starting_idx = (ego_idx + i * 10) % len(waypoints_xytheta)
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



# Initialize learner agent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

il_yaml_loc = '../il_utils/il_config.yaml'
il_config = yaml.load(open(il_yaml_loc), Loader=yaml.FullLoader)
agent = AgentPolicyMLP(il_config['policy_type']['agent']['observation_shape'],
                        il_config['policy_type']['agent']['hidden_dim'],
                        2,
                        il_config['policy_type']['agent']['learning_rate'],
                        device)

observation_shape = il_config['policy_type']['agent']['observation_shape']
downsampling_method = il_config['policy_type']['agent']['downsample_method']

# Load model weights
agent.load_state_dict(torch.load(model_path, map_location=device))


# Initialize lane switcher environment
map_name = 'General1'
lane_switcher_conf_path = \
    os.path.join('..', 'configs', map_name, 'lane_switcher_config.yaml')
with open(lane_switcher_conf_path) as file:
    lane_switcher_conf_dict = yaml.load(file, Loader=yaml.FullLoader)
lane_switcher_conf = Namespace(**lane_switcher_conf_dict)

expert_ego_planner = LaneSwitcher(conf=lane_switcher_conf)
opp_planner = LaneSwitcher(conf=lane_switcher_conf)
waypoints_xytheta = expert_ego_planner.lane_xytheta[0]

env = gym.make('f110_gym:f110-v0', map=lane_switcher_conf.map_path, map_ext='.png', num_agents=2)
render = LaneSwitcherRender(expert_ego_planner)
env.add_render_callback(render.render_callback)

overtake_scale_factor = 0.0
undesired_overtake_triggered = False


step_count = 0

for i in range(num_iter):
    opp_planner.collision_traj_overwritten_flag = None

    overtake_status = False
    start_perturb = False


    if undesired_overtake_behavior_prob > np.random.uniform():
        overtake_scale_factor = 1.5
        undesired_overtake_triggered = True
    else:
        overtake_scale_factor = 0.8
        undesired_overtake_triggered = False

    init_poses, _ = random_position(waypoints_xytheta, 2, rng, xy_noise_scale, theta_noise_scale)
    obs, _, done, _ = env.reset(init_poses)
    laptime = 0

    if render_sim:
        env.render('human')

    while not done and laptime < 8:
        oppo_pose = obsDict2oppoArray(obs, 0)
        # ego_best_traj = expert_ego_planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], oppo_pose,
        #                                  obs['linear_vels_x'][0])

        ## oppo planner here
        oppo_pose = obsDict2oppoArray(obs, 1)
        if laptime == 0:        # Make opponent always stay in one lane
            opp_best_traj = opp_planner.plan(obs['poses_x'][1], obs['poses_y'][1], obs['poses_theta'][1], oppo_pose,
                                         obs['linear_vels_x'][1], ego_oppo_collision_prob, collision_trigger_dist_thresh, 1)
        tracker_count = 0

        while not done and tracker_count < lane_switcher_conf.tracker_steps:
            # ego_steer, ego_speed = expert_ego_planner.tracker.plan(obs['poses_x'][0], obs['poses_y'][0],
            #                                                 obs['poses_theta'][0],
            #                                                 obs['linear_vels_x'][0], ego_best_traj)
            scan = agent_utils.downsample_and_extract_lidar(obs, 0, observation_shape, downsampling_method)
            agent_ego_action = agent.get_action(scan)
            #print("the type of scan is: ", type(scan))

            ## oppo planner here
            opp_steer, opp_speed = opp_planner.tracker.plan(obs['poses_x'][1], obs['poses_y'][1],
                                                            obs['poses_theta'][1],
                                                            obs['linear_vels_x'][1], opp_best_traj)
            ## oppo planner here
            action = np.array([agent_ego_action, [opp_steer, opp_speed*0.7]])
            obs, timestep, done, _ = env.step(action)

            step_count += 1

            if np.any(obs['collisions']):
                if (obs['collisions'][0] == 1.0) and (obs['collisions'][1] == 0.0):
                    num_ego_env_collision += 1
                if (obs['collisions'][0] == 1.0) and (obs['collisions'][1] == 1.0):
                    # print("Ego and opponent collide")
                    num_ego_oppo_collision += 1
                done = True
            
            oppo_scan = obs['scans'][1]
            oppo_x = obs['poses_x'][1]
            oppo_y = obs['poses_y'][1]
            oppo_theta = obs['poses_theta'][1]
            
            if overtake_status == False:
                overtake_status, left_oppo_scan_x, left_oppo_scan_y, right_oppo_scan_x, right_oppo_scan_y = utils.check_if_overtake(env, oppo_scan, oppo_x, oppo_y, oppo_theta)

            laptime += timestep
            tracker_count += 1
            if render_sim:
                env.render('human')
    # print("finish one episode")
    if overtake_status:
        num_overtake += 1

print("number of steps: ", step_count)
    #step_count = 0
print("=====================================")

collision_rate = num_ego_env_collision/num_iter
overtake_rate = num_overtake/num_iter

collision_rate, overtake_rate = utils.clip_res(collision_rate, overtake_rate)

print("Model path: ", model_path)
print("Percentage of iterations ego collide: ", collision_rate)
print("Percentage of iterations overtake: ", overtake_rate)