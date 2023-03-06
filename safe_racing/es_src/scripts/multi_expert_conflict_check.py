import torch
import gym
import numpy as np
import argparse
import yaml
from numpy.linalg import norm

import utils

from il_utils.policies.agents.agent_mlp import AgentPolicyMLP
from il_utils.policies.experts.expert_waypoint_follower import ExpertWaypointFollower

import il_utils.utils.env_utils as env_utils

# hg-dagger import
from il_utils.dataset import Dataset
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
import random

# General Settings
render_sim = False

online_ego_oppo_collision_check = False
ego_oppo_collision_prob = 0.0 # Probability of collision between ego vehicle and opponent vehicle
collision_trigger_dist_thresh = 5 # Distance threshold for triggering collision
undesired_overtake_behavior_prob = 0.5 # Probability of high speed overtaking
cbf_triggered_steps_to_remove = 70

# Saving settings
use_CBF_filter = True
save_folder = 'multi_weak_expert_conflict_check'

# use_CBF_filter = False
# save_folder = 'undesired_overtake_behavior_prob/1.0/no_filter'

save_skip_interval = 10
# save_skip_interval = 100




# seed and iter
expert_seeds = [5392, 6313, 5065, 7682, 4864] # Generate by random.randrange(0, 10000, 1), executed for 5 times
num_expert = len(expert_seeds)

n_iter_each_expert = 210

#n_iter = 1010
#seed = 6300
#np.random.seed(seed)
#rng = np.random.default_rng(seed)

cos_sim_threshold = 0.99


xy_noise_scale = 0.0
theta_noise_scale = 0.0


global dataset
dataset = Dataset()

global conflict_resolution_log
conflict_resolution_log = {'conflict_idx': [],
                           'lidar': [],
                           'pose_x': [],
                           'pose_y': [],
                           'pose_theta': [],
                           'action_steer': [],
                           'action_speed': [],
                           'cbf_val': []
                           }

global curr_conflict_idx
curr_conflict_idx = 0

total_steps = 0

def cosine_similarity(A, B):
    return np.dot(A,B)/(norm(A, axis=1)*norm(B))


def conflict_check_scoring_and_resolution(pose_x, pose_y, pose_theta, curr_scan, expert_action, curr_vel_x, curr_vel_y, CBF_value, global_dataset, similarity_threshold=0.9995, alpha=0.5):
    global dataset
    global conflict_resolution_log
    global curr_conflict_idx
    
    global_dataset = dataset

    global_scans = global_dataset.scans

    

    if global_scans is not None:
        cos_sim_arr = cosine_similarity(global_scans, curr_scan)
    else:
        return

    indices_above_sim_thresh = np.where(cos_sim_arr > similarity_threshold)[0]

    log_conflict_resolution = False

    if indices_above_sim_thresh.size > 0:
        curr_conflict_idx = curr_conflict_idx + 1
        log_conflict_resolution = True

    #print("cos_sim_arr: ", cos_sim_arr)
    #print("indices_above_sim_thresh: ", indices_above_sim_thresh)

    #       1. Store CBF value in the global dataset
    #       2. Calculate speed from curr_vel_x, curr_vel_y
    curr_speed = np.sqrt(curr_vel_x**2 + curr_vel_y**2)
    #       3. Put all selected similar relevant data entries (speed, CBF_value) together
    data_entries_above_thresh = {'idx': [],
                                 'speed': [],
                                 'CBF_value': [],
                                 'normalized_speed': [],
                                 'normalized_CBF_value': [],
                                 'score': []}
    if log_conflict_resolution:
        conflict_resolution_log['conflict_idx'].append(curr_conflict_idx)
        conflict_resolution_log['lidar'].append(curr_scan)
        conflict_resolution_log['pose_x'].append(pose_x)
        conflict_resolution_log['pose_y'].append(pose_y)
        conflict_resolution_log['pose_theta'].append(pose_theta)
        conflict_resolution_log['action_steer'].append(expert_action[0])
        conflict_resolution_log['action_speed'].append(expert_action[1])
        conflict_resolution_log['cbf_val'].append(CBF_value)

    for idx1 in indices_above_sim_thresh:
        data_entries_above_thresh['idx'].append(idx1)
        curr_entry_speed = np.sqrt(global_dataset.vel_x[idx1]**2 + global_dataset.vel_y[idx1]**2)
        data_entries_above_thresh['speed'].append(curr_entry_speed)
        data_entries_above_thresh['CBF_value'].append(global_dataset.cbf_val[idx1])

        if log_conflict_resolution:
            conflict_resolution_log['conflict_idx'].append(curr_conflict_idx)
            conflict_resolution_log['lidar'].append(global_scans[idx1])
            conflict_resolution_log['pose_x'].append(global_dataset.poses_x[idx1])
            conflict_resolution_log['pose_y'].append(global_dataset.poses_y[idx1])
            conflict_resolution_log['pose_theta'].append(global_dataset.poses_theta[idx1])
            conflict_resolution_log['action_steer'].append(global_dataset.actions[idx1][0])
            conflict_resolution_log['action_speed'].append(global_dataset.actions[idx1][1])
            conflict_resolution_log['cbf_val'].append(global_dataset.cbf_val[idx1])



    #       4. Calculate the normalized speed and CBF_value
    speed_concat = np.array(data_entries_above_thresh['speed'])
    speed_concat = np.append(speed_concat, curr_speed)
    speed_norm = np.linalg.norm(speed_concat)
    CBF_value_concat = np.array(data_entries_above_thresh['CBF_value'])
    CBF_value_concat = np.append(CBF_value_concat, CBF_value)
    CBF_value_norm = np.linalg.norm(CBF_value_concat)

    #       5. Add normalized speed and CBF_value together with a weighting factor.
    for idx2 in range(len(data_entries_above_thresh['idx'])):
        curr_entry_normalized_speed = data_entries_above_thresh['speed'][idx2] / speed_norm
        curr_entry_normalized_CBF_value = data_entries_above_thresh['CBF_value'][idx2] / CBF_value_norm
        curr_entry_score = alpha * curr_entry_normalized_speed + (1 - alpha) * curr_entry_normalized_CBF_value
        data_entries_above_thresh['normalized_speed'].append(curr_entry_normalized_speed)
        data_entries_above_thresh['normalized_CBF_value'].append(curr_entry_normalized_CBF_value)
        data_entries_above_thresh['score'].append(curr_entry_score)

    #       6. Compare which one is larger. The entry with larger score will be used to replace the lower entry
    curr_normalized_speed = curr_speed / speed_norm
    curr_normalized_CBF_value = CBF_value / CBF_value_norm
    curr_score = alpha * curr_normalized_speed + (1 - alpha) * curr_normalized_CBF_value
    if_curr_score_larger = all(x < curr_score for x in data_entries_above_thresh['score'])
    if if_curr_score_larger:
        #           7. If the current entry is larger, replace the lower entry with the current entry
        #           8. Update the global dataset
        for idx3 in data_entries_above_thresh['idx']:
            global_dataset.vel_x[idx3] = curr_vel_x
            global_dataset.vel_y[idx3] = curr_vel_y
            global_dataset.cbf_val[idx3] = CBF_value
            global_dataset.actions[idx3] = expert_action
        
    dataset = global_dataset

    return 


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

iter = 0

for curr_expert_idx in range(num_expert):
    curr_expert_seed = expert_seeds[curr_expert_idx]
    np.random.seed(curr_expert_seed)
    rng = np.random.default_rng(curr_expert_seed)
    random.seed(curr_expert_seed)
    torch.manual_seed(curr_expert_seed)
    torch.cuda.manual_seed(curr_expert_seed)

    for curr_iter_idx_under_curr_expert in range(n_iter_each_expert):
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

        traj = {"poses_x": [], "poses_y": [], "poses_theta": [], "scans": [], "actions": [], "cbf_val": [], "vel_x": [], "vel_y": []}
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
                    traj["vel_x"].append(obs["linear_vels_x"][0])
                    traj["vel_y"].append(obs["linear_vels_y"][0])
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
                    
                    CBF_value_border = 999
                    if use_CBF_filter:
                        
                        if online_ego_oppo_collision_check:
                            # online monitoring and filtering for agent-agent collision
                            if step_num >= 1:
                                x_ego_next, y_ego_next = obs["poses_x"][0], obs["poses_y"][0]
                                x_opp_next, y_opp_next = obs["poses_x"][1], obs["poses_y"][1]
                                res, CBF_value = CBFagent(x_ego, y_ego, x_opp, y_opp, x_ego_next, y_ego_next, x_opp_next, y_opp_next)

                                curr_vel_x = obs["linear_vels_x"][0]
                                curr_vel_y = obs["linear_vels_y"][0]
                                conflict_check_scoring_and_resolution(obs["poses_x"][0], obs["poses_y"][0], obs["poses_theta"][0], scan, expert_ego_action, curr_vel_x, curr_vel_y, CBF_value, dataset)

                                if res == False:
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


                            res_border, CBF_value_border = CBFborder(x_ego_left, y_ego_left, x_line, y_line, x_ego_next_left, y_ego_next_left, x_line_next, y_line_next)

                            curr_vel_x = obs["linear_vels_x"][0]
                            curr_vel_y = obs["linear_vels_y"][0]
                            conflict_check_scoring_and_resolution(obs["poses_x"][0], obs["poses_y"][0], obs["poses_theta"][0], scan, expert_ego_action, curr_vel_x, curr_vel_y, CBF_value_border, dataset)
                            
                            if left_oppo_hit_check == False and res_border == False:
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


                            res_border, CBF_value_border = CBFborder(x_ego_left, y_ego_left, x_line, y_line, x_ego_next_left, y_ego_next_left, x_line_next, y_line_next)

                            curr_vel_x = obs["linear_vels_x"][0]
                            curr_vel_y = obs["linear_vels_y"][0]
                            conflict_check_scoring_and_resolution(obs["poses_x"][0], obs["poses_y"][0], obs["poses_theta"][0], scan, expert_ego_action, curr_vel_x, curr_vel_y, CBF_value_border, dataset)                     
                            
                            if right_oppo_hit_check == False and res_border == False:
                                CBF_collision_flag = True
                            x_ego_right, y_ego_right, x_line1, y_line1 = x_ego_next_right, y_ego_next_right, x_line_next1, y_line_next1
                            #if right_oppo_hit_check:
                            #    break
                        else:
                            x_ego_right, y_ego_right = obs["poses_x"][0], obs["poses_y"][0]
                            x_line1, y_line1 = right_pt_x, right_pt_y
                    traj["cbf_val"].append(CBF_value_border)    

                else:
                    print("Learner agent is in control")
                    curr_ego_action = agent_ego_action      

                action = np.array([curr_ego_action, curr_oppo_action])
                obs, timestep, done, _ = env.step(action)
                step_num += 1

                total_steps += 1

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
                    traj["cbf_val"], _ = simple_filtering(traj["cbf_val"], cbf_triggered_steps_to_remove)
                    traj['vel_x'], _ = simple_filtering(traj['vel_x'], cbf_triggered_steps_to_remove)
                    traj['vel_y'], _ = simple_filtering(traj['vel_y'], cbf_triggered_steps_to_remove)
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
        if (traj["poses_x"] is None) or (traj["poses_y"] is None) or (traj["poses_theta"] is None) or (traj["scans"] is None) or (traj["actions"] is None) or (traj["cbf_val"] is None):
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
        traj["cbf_val"] = np.vstack(traj["cbf_val"])
        traj["vel_x"] = np.vstack(traj["vel_x"])
        traj["vel_y"] = np.vstack(traj["vel_y"])
        #iter += 1
        
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
                res, CBF_value = CBFagent(x_ego, y_ego, x_opp, y_opp, x_ego_next, y_ego_next, x_opp_next, y_opp_next)
                if res == False:
                    traj["poses_x"] = traj["poses_x"][0: i - 1]
                    traj["poses_y"] = traj["poses_y"][0: i - 1]
                    traj["poses_theta"] = traj["poses_theta"][0: i - 1]
                    traj["scans"] = traj["scans"][0: i - 1]
                    traj["actions"] = traj["actions"][0: i - 1]
                    traj["cbf_val"] = traj["cbf_val"][0: i - 1]
                    traj["vel_x"] = traj["vel_x"][0: i - 1]
                    traj["vel_y"] = traj["vel_y"][0: i - 1]
                    print("original trajectory length: ", trajLen)
                    print("current trajectory length (after filtering): ", i)
                    break

        print("Adding to dataset...")
        print("CBF length: ", len(traj["cbf_val"]))
        print("poses_x length: ", len(traj["poses_x"]))
        dataset.add(traj)

        # log['Number of Samples'].append(dataset.get_num_of_total_samples())
        # log['Number of Expert Queries'].append(dataset.get_num_of_total_samples())

        # Train the agent
        iter = curr_expert_idx * n_iter_each_expert + curr_iter_idx_under_curr_expert
        print("Training the agent...")
        for _ in range(n_batch_updates_per_iter):
            train_batch = dataset.sample(train_batch_size)
            agent.train(train_batch["scans"], train_batch["actions"])
        
        if (iter % save_skip_interval == 0) and (iter > 0):
            print("Save the current model")
            model_path = Path('models/'+ save_folder + f'/iter_{int(iter)}_model.pkl')
            # model_path = Path('models/'+ save_folder + f'/iter_{int(iter)}_model.pkl')
            model_path.parent.mkdir(parents=True, exist_ok=True) 
            torch.save(agent.state_dict(), model_path)

            if use_CBF_filter:
                df = pd.DataFrame(data_removal_log)
                log_path = Path('models/'+ save_folder + f'/data_log/data_log_{int(iter)}.csv')
                log_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(log_path, index=False)
            
            conflict_resolution_df = pd.DataFrame(conflict_resolution_log)
            conflict_resolution_log_path = Path('models/'+ save_folder + f'/conflict_resolution_log/conflict_resolution_log_{int(iter)}.csv')
            conflict_resolution_log_path.parent.mkdir(parents=True, exist_ok=True)
            conflict_resolution_df.to_csv(conflict_resolution_log_path, index=False)
        
        # agent_utils.save_log_and_model(log, agent, save_folder)
print("total number of steps: ", total_steps)
        
            
