import torch
import gym
import numpy as np
import argparse
import yaml

import utils

from il_utils.policies.agents.agent_mlp import AgentPolicyMLP
from il_utils.policies.experts.expert_waypoint_follower import ExpertWaypointFollower

import il_utils.utils.env_utils as env_utils

# hg-dagger import
from il_utils.dataset_old import Dataset
from il_utils.utils import agent_utils
from pathlib import Path
# from realtime_raceline_coordinate import get_realtime_raceline_coordinate
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
from CBF_filter import CBFagent, CBFborder

import pandas as pd

# General Settings
render_sim = False

online_ego_oppo_collision_check = False
ego_oppo_collision_prob = 0.0 # Probability of collision between ego vehicle and opponent vehicle
collision_trigger_dist_thresh = 5 # Distance threshold for triggering collision
undesired_overtake_behavior_prob = 0.1 # Probability of high speed overtaking
cbf_triggered_steps_to_remove = 70

# Saving settings
use_CBF_filter = True
save_folder = 'fixed_training_dataset_length_comparison/0.1'

# use_CBF_filter = False
# save_folder = 'undesired_overtake_behavior_prob/1.0/no_filter'

save_skip_interval = 10
# save_skip_interval = 100

n_iter = 1010


# seed
seed = 6300
rng = np.random.default_rng(seed)
xy_noise_scale = 0.0
theta_noise_scale = 0.0

def random_position(waypoints_xytheta, sampled_number=1, rng=None, xy_noise=0.0, theta_noise=0.0):
    ego_idx = rng.choice(np.arange(0, len(waypoints_xytheta)), 1)[0]
    # ego_idx = 790
    print(f'ego_idx is {ego_idx}')
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


def simple_filtering(data_list, filtered_steps):
    num_steps_removed = 0
    if len(data_list) > filtered_steps:
        new_data_list = data_list[:len(data_list) - filtered_steps]
        num_steps_removed = filtered_steps
    else:
        num_steps_removed = len(data_list)
        new_data_list = None
    return new_data_list, num_steps_removed


# Initialize learner agent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

il_yaml_loc = '../il_utils/il_config.yaml'
il_config = yaml.load(open(il_yaml_loc), Loader=yaml.FullLoader)
agent = AgentPolicyMLP(il_config['policy_type']['agent']['observation_shape'],
                        il_config['policy_type']['agent']['hidden_dim'],
                        2,
                        il_config['policy_type']['agent']['learning_rate'],
                        device)

agent_comp = AgentPolicyMLP(il_config['policy_type']['agent']['observation_shape'],
                        il_config['policy_type']['agent']['hidden_dim'],
                        2,
                        il_config['policy_type']['agent']['learning_rate'],
                        device)

observation_shape = il_config['policy_type']['agent']['observation_shape']
downsampling_method = il_config['policy_type']['agent']['downsample_method']



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





train_batch_size = 64
n_batch_updates_per_iter = 1000

dataset = Dataset()

dataset_comp = Dataset()

log = {'Number of Samples': [],
        'Number of Expert Queries': [],
        'Mean Distance Travelled': [],
        'STDEV Distance Travelled': [],
        'Mean Reward': [],
        'STDEV Reward': []}

data_removal_log = {'Iteration': [],
                    'Total Steps Before Removal': [],
                    'Total Steps After Removal': [],
                    'Number of Steps Removed': [],
                    'Ratio of Steps Removed': []}

overtake_scale_factor = 0.0
undesired_overtake_triggered = False

for iter in range(n_iter):
    data_removal_log['Iteration'].append(iter)

    CBF_collision_flag = False
    step_num = 0
    step_num_left = 0
    step_num_right = 0
    # Reset peer collision overwritten flag
    expert_ego_planner.collision_traj_overwritten_flag = None
    opp_planner.collision_traj_overwritten_flag = None

    overtake_status = False
    start_perturb = False

    # Trigger undesired overtaking behavior by chance
    if undesired_overtake_behavior_prob > np.random.uniform():
        overtake_scale_factor = 1.5
        undesired_overtake_triggered = True
    else:
        overtake_scale_factor = 0.8
        undesired_overtake_triggered = False

    print("\niter {}:".format(iter))

    ego_collided = False

    init_poses, _ = random_position(waypoints_xytheta, 2, rng, xy_noise_scale, theta_noise_scale)
    obs, _, done, _ = env.reset(init_poses)
    laptime = 0

    if render_sim:
        env.render('human_fast')

    traj = {"poses_x": [], "poses_y": [], "poses_theta": [], "scans": [], "actions": []}
    traj_comp = {"poses_x": [], "poses_y": [], "poses_theta": [], "scans": [], "actions": []}
    oppo_traj = {"poses_x": [], "poses_y": [], "poses_theta": [], "scans": [], "actions": []}

    # Bootstrap the training using BC first
    while not done and laptime < 8:
        oppo_pose = obsDict2oppoArray(obs, 0)
        ego_best_traj = expert_ego_planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], oppo_pose,
                                         obs['linear_vels_x'][0], ego_oppo_collision_prob, collision_trigger_dist_thresh, overtake_scale_factor)
        ## oppo planner here
        oppo_pose = obsDict2oppoArray(obs, 1)
        if laptime == 0:        # Make opponent always stay in one lane
            opp_best_traj = opp_planner.plan(obs['poses_x'][1], obs['poses_y'][1], obs['poses_theta'][1], oppo_pose,
                                         obs['linear_vels_x'][1], ego_oppo_collision_prob, collision_trigger_dist_thresh, overtake_scale_factor)
        tracker_count = 0

        while not done and tracker_count < lane_switcher_conf.tracker_steps:
            # Expert ego planner
            expert_ego_steer, expert_ego_speed = expert_ego_planner.tracker.plan(obs['poses_x'][0], obs['poses_y'][0],
                                                            obs['poses_theta'][0],
                                                            obs['linear_vels_x'][0], ego_best_traj)
            if start_perturb:
                expert_ego_action = [-expert_ego_steer, expert_ego_speed]
            else:
                expert_ego_action = [expert_ego_steer, expert_ego_speed]

            ego_scan = obs['scans'][0]
            ego_x = obs['poses_x'][0]
            ego_y = obs['poses_y'][0]
            ego_theta = obs['poses_theta'][0]
            
            left_pt_x, left_pt_y, right_pt_x, right_pt_y, closest_left_dist, closest_right_dist = utils.get_closest_boundary_pts(env, ego_scan, ego_x, ego_y, ego_theta)
            front_closest_pt_x, front_closest_pt_y, front_closest_dist = utils.get_front_closest_pt(ego_scan, ego_x, ego_y, ego_theta)

            if undesired_overtake_triggered:
                if utils.check_target_pt_hit_oppo(env, left_pt_x, left_pt_y) or utils.check_target_pt_hit_oppo(env, right_pt_x, right_pt_y):
                    start_perturb = True

            # Agent ego policy
            scan = agent_utils.downsample_and_extract_lidar(obs, 0, observation_shape, downsampling_method)
            # scan = obs["scans"][0]
            # print(obs["scans"][0])
            agent_ego_action = agent.get_action(scan)

            agent_ego_steer = agent_ego_action[0]
            agent_ego_speed = agent_ego_action[1]

            ## oppo planner here
            opp_steer, opp_speed = opp_planner.tracker.plan(obs['poses_x'][1], obs['poses_y'][1],
                                                            obs['poses_theta'][1],
                                                            obs['linear_vels_x'][1], opp_best_traj)
            curr_oppo_action = [opp_steer, opp_speed*0.7]
            

            # Decide if agent or expert has control
            if (np.abs(agent_ego_steer - expert_ego_steer) > 0.1) or (np.abs(agent_ego_speed - expert_ego_speed) > 1):
                print("Expert is in control")
                curr_ego_action = expert_ego_action

                # Append to current ego trajectory
                traj["scans"].append(scan)
                traj["poses_x"].append(obs["poses_x"][0])
                traj["poses_y"].append(obs["poses_y"][0])
                traj["poses_theta"].append(obs["poses_theta"][0])
                traj["actions"].append(expert_ego_action)

                traj_comp["scans"].append(scan)
                traj_comp["poses_x"].append(obs["poses_x"][0])
                traj_comp["poses_y"].append(obs["poses_y"][0])
                traj_comp["poses_theta"].append(obs["poses_theta"][0])
                traj_comp["actions"].append(expert_ego_action)
                #track_point0, track_point1 = get_realtime_raceline_coordinate(lane_switcher_conf, [obs["poses_x"][0], obs["poses_y"][0], obs["poses_theta"][0]], dist_threshold=4)
                #print("ego position: ", obs["poses_x"][0], obs["poses_y"][0])
                #print("boundary point 1: ", track_point0)
                #print("boundary point 2: ", track_point1)
                #print("lane width: ", ((track_point0[0] - track_point1[0]) ** 2 + (track_point0[1] - track_point1[1]) ** 2) ** (1/2))
                #print("center line point: ", (track_point0 + track_point1) / 2)
                
                # Append to current oppo trajectory
                oppo_traj["scans"].append(agent_utils.downsample_and_extract_lidar(obs, 1, observation_shape, downsampling_method))
                oppo_traj["poses_x"].append(obs["poses_x"][1])
                oppo_traj["poses_y"].append(obs["poses_y"][1])
                oppo_traj["poses_theta"].append(obs["poses_theta"][1])
                oppo_traj["actions"].append(curr_oppo_action)
                
                if use_CBF_filter:
                    if online_ego_oppo_collision_check:
                        # online monitoring and filtering for agent-agent collision
                        if step_num >= 1:
                            x_ego_next, y_ego_next = obs["poses_x"][0], obs["poses_y"][0]
                            x_opp_next, y_opp_next = obs["poses_x"][1], obs["poses_y"][1]
                            if CBFagent(x_ego, y_ego, x_opp, y_opp, x_ego_next, y_ego_next, x_opp_next, y_opp_next) == False:
                                CBF_collision_flag = True
                            x_ego, y_ego, x_opp, y_opp = x_ego_next, y_ego_next, x_opp_next, y_opp_next
                        else:
                            x_ego, y_ego = obs["poses_x"][0], obs["poses_y"][0]
                            x_opp, y_opp = obs["poses_x"][1], obs["poses_y"][1]
                    
                    # online monitoring and filtering for agent-boundary1 collision
                    

                    left_pt_x, left_pt_y, right_pt_x, right_pt_y, closest_left_dist, closest_right_dist = utils.get_closest_boundary_pts(env, scan, obs["poses_x"][0], obs["poses_y"][0], obs["poses_theta"][0])
                    front_closest_pt_x, front_closest_pt_y, front_closest_dist = utils.get_front_closest_pt(scan, obs["poses_x"][0], obs["poses_y"][0], obs["poses_theta"][0])
                    
                    # Test for point alignment checking
                    left_oppo_hit_check = utils.check_target_pt_hit_oppo(env, left_pt_x, left_pt_y)
                    right_oppo_hit_check = utils.check_target_pt_hit_oppo(env, right_pt_x, right_pt_y)
                    front_oppo_hit_check = utils.check_target_pt_hit_oppo(env, front_closest_pt_x, front_closest_pt_y)
                    print(left_oppo_hit_check, right_oppo_hit_check)
                    if left_oppo_hit_check:
                        print('Left point hit opponent!!!!!')
                    
                    if right_oppo_hit_check:
                        print('Right point hit opponent!!!!!')
                    
                    if front_oppo_hit_check:
                        print('Front point hit opponent!!!!!')

                    if step_num_left >= 1:
                        x_ego_next_left, y_ego_next_left = obs["poses_x"][0], obs["poses_y"][0]
                        x_line_next, y_line_next = left_pt_x, left_pt_y
                        if left_oppo_hit_check == False and CBFborder(x_ego_left, y_ego_left, x_line, y_line, x_ego_next_left, y_ego_next_left, x_line_next, y_line_next) == False:
                            CBF_collision_flag = True
                        x_ego_left, y_ego_left, x_line, y_line = x_ego_next_left, y_ego_next_left, x_line_next, y_line_next
                        #if left_oppo_hit_check:
                        #    break
                    else:
                        x_ego_left, y_ego_left = obs["poses_x"][0], obs["poses_y"][0]
                        x_line, y_line = left_pt_x, left_pt_y
                    
                    # online monitoring and filtering for agent-boundary2 collision
                    if step_num_right >= 1:
                        x_ego_next_right, y_ego_next_right = obs["poses_x"][0], obs["poses_y"][0]
                        x_line_next1, y_line_next1 = right_pt_x, right_pt_y
                        if right_oppo_hit_check == False and CBFborder(x_ego_right, y_ego_right, x_line1, y_line1, x_ego_next_right, y_ego_next_right, x_line_next1, y_line_next1) == False:
                            CBF_collision_flag = True
                        x_ego_right, y_ego_right, x_line1, y_line1 = x_ego_next_right, y_ego_next_right, x_line_next1, y_line_next1
                        #if right_oppo_hit_check:
                        #    break
                    else:
                        x_ego_right, y_ego_right = obs["poses_x"][0], obs["poses_y"][0]
                        x_line1, y_line1 = right_pt_x, right_pt_y

            else:
                print("Learner agent is in control")
                curr_ego_action = agent_ego_action      

            action = np.array([curr_ego_action, curr_oppo_action])
            obs, timestep, done, _ = env.step(action)
            step_num += 1

            if use_CBF_filter:
                if right_oppo_hit_check:
                    step_num_right = 0
                else:
                    step_num_right += 1
                if left_oppo_hit_check:
                    step_num_left = 0
                else:
                    step_num_left += 1

            # print(action)
            if np.any(obs['collisions']) or CBF_collision_flag:
                done = True

                data_removal_log["Total Steps Before Removal"].append(len(traj["scans"]))

                traj["scans"], num_steps_removed = simple_filtering(traj["scans"], cbf_triggered_steps_to_remove)
                traj["poses_x"], _ = simple_filtering(traj["poses_x"], cbf_triggered_steps_to_remove)
                traj["poses_y"], _ = simple_filtering(traj["poses_y"], cbf_triggered_steps_to_remove)
                traj["poses_theta"], _ = simple_filtering(traj["poses_theta"], cbf_triggered_steps_to_remove)
                traj["actions"], _ = simple_filtering(traj["actions"], cbf_triggered_steps_to_remove)
                oppo_traj["scans"], _ = simple_filtering(oppo_traj["scans"], cbf_triggered_steps_to_remove)
                oppo_traj["poses_x"], _ = simple_filtering(oppo_traj["poses_x"], cbf_triggered_steps_to_remove)
                oppo_traj["poses_y"], _ = simple_filtering(oppo_traj["poses_y"], cbf_triggered_steps_to_remove)
                oppo_traj["poses_theta"], _ = simple_filtering(oppo_traj["poses_theta"], cbf_triggered_steps_to_remove)
                oppo_traj["actions"], _ = simple_filtering(oppo_traj["actions"], cbf_triggered_steps_to_remove)

                data_removal_log["Number of Steps Removed"].append(num_steps_removed)
                if traj["scans"] != None:
                    data_removal_log["Total Steps After Removal"].append(len(traj["scans"]))
                    data_removal_log["Ratio of Steps Removed"].append(num_steps_removed / (len(traj["scans"])+num_steps_removed))
                else:
                    data_removal_log["Total Steps After Removal"].append(0)
                    data_removal_log["Ratio of Steps Removed"].append(1)


            laptime += timestep
            tracker_count += 1
            if render_sim:
                env.render('human_fast')

    # If all data are filtered, skip this iteration
    if (traj["poses_x"] is None) or (traj["poses_y"] is None) or (traj["poses_theta"] is None) or (traj["scans"] is None) or (traj["actions"] is None):
        continue
    
    if not done:
        data_removal_log["Total Steps Before Removal"].append(len(traj["scans"]))
        data_removal_log["Number of Steps Removed"].append(0)
        data_removal_log["Total Steps After Removal"].append(len(traj["scans"]))
        data_removal_log["Ratio of Steps Removed"].append(0)


    # Add trajectory to dataset
    traj["poses_x"] = np.vstack(traj["poses_x"])
    traj["poses_y"] = np.vstack(traj["poses_y"])
    traj["poses_theta"] = np.vstack(traj["poses_theta"])
    traj["scans"] = np.vstack(traj["scans"])
    traj["actions"] = np.vstack(traj["actions"])

    traj_comp["poses_x"] = np.vstack(traj_comp["poses_x"])
    traj_comp["poses_y"] = np.vstack(traj_comp["poses_y"])
    traj_comp["poses_theta"] = np.vstack(traj_comp["poses_theta"])
    traj_comp["scans"] = np.vstack(traj_comp["scans"])
    traj_comp["actions"] = np.vstack(traj_comp["actions"])
    
    oppo_traj["poses_x"] = np.vstack(oppo_traj["poses_x"])
    oppo_traj["poses_y"] = np.vstack(oppo_traj["poses_y"])
    oppo_traj["poses_theta"] = np.vstack(oppo_traj["poses_theta"])
    oppo_traj["scans"] = np.vstack(oppo_traj["scans"])
    oppo_traj["actions"] = np.vstack(oppo_traj["actions"])

    if use_CBF_filter and (not online_ego_oppo_collision_check):
        print("Checking agent-agent safety of trajectory using CBF...")
        
        # offline filter unsafe traj with agent-agent collision
        
        trajLen = traj["poses_x"].shape[0]
        for i in range(trajLen - 1):
            x_ego = traj["poses_x"][i, 0]
            y_ego = traj["poses_y"][i, 0]
            x_ego_next = traj["poses_x"][i + 1, 0]
            y_ego_next = traj["poses_y"][i + 1, 0]
            x_opp = oppo_traj["poses_x"][i, 0]
            y_opp = oppo_traj["poses_y"][i, 0]
            x_opp_next = oppo_traj["poses_x"][i + 1, 0]
            y_opp_next = oppo_traj["poses_y"][i + 1, 0]
            # when CBFagent returns false, it means that a collision might happen, so the rest of dataset need to be deleted (or assign a low weight while training).
            if CBFagent(x_ego, y_ego, x_opp, y_opp, x_ego_next, y_ego_next, x_opp_next, y_opp_next) == False:
                traj["poses_x"] = traj["poses_x"][0: i - 1]
                traj["poses_y"] = traj["poses_y"][0: i - 1]
                traj["poses_theta"] = traj["poses_theta"][0: i - 1]
                traj["scans"] = traj["scans"][0: i - 1]
                traj["actions"] = traj["actions"][0: i - 1]
                print("original trajectory length: ", trajLen)
                print("current trajectory length (after filtering): ", i)
                break

    print("Adding to dataset...")
    dataset.add(traj)
    dataset_comp.add(traj_comp)

    # log['Number of Samples'].append(dataset.get_num_of_total_samples())
    # log['Number of Expert Queries'].append(dataset.get_num_of_total_samples())

    # Train the agent
    print("Training the agent...")
    for _ in range(n_batch_updates_per_iter):
        train_batch = dataset.sample(train_batch_size)
        agent.train(train_batch["scans"], train_batch["actions"])
    
    # Randomly truncate dataset_comp to the same length as dataset
    if dataset_comp.get_num_of_total_samples() > dataset.get_num_of_total_samples():
        dataset_comp.random_truncate(dataset.get_num_of_total_samples())
    
    for _comp in range(n_batch_updates_per_iter):
        train_batch_comp = dataset_comp.sample(train_batch_size)
        agent_comp.train(train_batch_comp["scans"], train_batch_comp["actions"])
    
    if (iter % save_skip_interval == 0) and (iter > 0):
        print("Save the current model")
        model_path = Path('models/'+ save_folder + f'/cbf_filtered/iter_{int(iter)}_model.pkl')
        # model_path = Path('models/'+ save_folder + f'/iter_{int(iter)}_model.pkl')
        model_path.parent.mkdir(parents=True, exist_ok=True) 
        torch.save(agent.state_dict(), model_path)

        model_path_comp = Path('models/'+ save_folder + f'/random_truncate/iter_{int(iter)}_model_comp.pkl')
        # model_path_comp = Path('models/'+ save_folder + f'/iter_{int(iter)}_model_comp.pkl')
        model_path_comp.parent.mkdir(parents=True, exist_ok=True)
        torch.save(agent_comp.state_dict(), model_path_comp)

        if use_CBF_filter:
            df = pd.DataFrame(data_removal_log)
            log_path = Path('models/'+ save_folder + f'/data_log/data_log_{int(iter)}.csv')
            log_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(log_path, index=False)

            # Save datasets as npz
            dataset_path = Path('models/'+ save_folder + f'/dataset_cbf/iter_{int(iter)}_dataset_cbf.npz')
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            dataset.save_npz(dataset_path)

            dataset_path_comp = Path('models/'+ save_folder + f'/dataset_random/iter_{int(iter)}_dataset_random.npz')
            dataset_path_comp.parent.mkdir(parents=True, exist_ok=True)
            dataset_comp.save_npz(dataset_path_comp)
    
    # agent_utils.save_log_and_model(log, agent, save_folder)
            
