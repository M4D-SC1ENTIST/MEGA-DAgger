import utils


import os
import json
import numpy as np
from argparse import Namespace

# Set planner
from es.planner.lattice_planner import LatticePlanner
from es.planner.lane_switcher import LaneSwitcher
import gym
from es.utils.DataProcessor import RolloutDataLogger
from es.utils.utils import *
from es.utils.visualize import LatticePlannerRender, LaneSwitcherRender
from es.worker import calculate_objectives
import yaml

# Settings for testing the expert performance
render_sim = True
ego_oppo_collision_prob = 0.0 # Probability of collision between ego vehicle and opponent vehicle
collision_trigger_dist_thresh = 5 # Distance threshold for triggering collision
high_speed_overtake_prob = 0.0 # Probability of high speed overtaking

# Seed
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


map_name = 'General1'
lane_switcher_conf_path = \
    os.path.join('..', 'configs', map_name, 'lane_switcher_config.yaml')
with open(lane_switcher_conf_path) as file:
    lane_switcher_conf_dict = yaml.load(file, Loader=yaml.FullLoader)
lane_switcher_conf = Namespace(**lane_switcher_conf_dict)

ego_planner = LaneSwitcher(conf=lane_switcher_conf)
opp_planner = LaneSwitcher(conf=lane_switcher_conf)
waypoints_xytheta = ego_planner.lane_xytheta[0]

env = gym.make('f110_gym:f110-v0', map=lane_switcher_conf.map_path, map_ext='.png', num_agents=2)
render = LaneSwitcherRender(ego_planner)
env.add_render_callback(render.render_callback)


for i in range(10):
    # Reset peer collision overwritten flag
    ego_planner.collision_traj_overwritten_flag = None
    opp_planner.collision_traj_overwritten_flag = None


    init_poses, _ = random_position(waypoints_xytheta, 2, rng, xy_noise_scale, theta_noise_scale)
    obs, _, done, _ = env.reset(init_poses)
    laptime = 0

    # Set overtake high speed scale factor
    if high_speed_overtake_prob < np.random.uniform():
        overtake_scale_factor = 0.8
    else:
        overtake_scale_factor = 1.5

    if render_sim:
        env.render('human')

    while not done and laptime < 8:
        oppo_pose = obsDict2oppoArray(obs, 0)
        ego_best_traj = ego_planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], oppo_pose,
                                         obs['linear_vels_x'][0], ego_oppo_collision_prob, collision_trigger_dist_thresh, overtake_scale_factor)

        ## oppo planner here
        oppo_pose = obsDict2oppoArray(obs, 1)
        if laptime == 0:        # Make opponent always stay in one lane
            opp_best_traj = opp_planner.plan(obs['poses_x'][1], obs['poses_y'][1], obs['poses_theta'][1], oppo_pose,
                                         obs['linear_vels_x'][1], ego_oppo_collision_prob, collision_trigger_dist_thresh, overtake_scale_factor)
        tracker_count = 0

        while not done and tracker_count < lane_switcher_conf.tracker_steps:
            ego_steer, ego_speed = ego_planner.tracker.plan(obs['poses_x'][0], obs['poses_y'][0],
                                                            obs['poses_theta'][0],
                                                            obs['linear_vels_x'][0], ego_best_traj)
            ## oppo planner here
            opp_steer, opp_speed = opp_planner.tracker.plan(obs['poses_x'][1], obs['poses_y'][1],
                                                            obs['poses_theta'][1],
                                                            obs['linear_vels_x'][1], opp_best_traj)
            ## oppo planner here
            action = np.array([[ego_steer, ego_speed], [opp_steer, opp_speed*0.8]])

            ego_scan = obs['scans'][0]

            ego_x = obs['poses_x'][0]
            ego_y = obs['poses_y'][0]
            ego_theta = obs['poses_theta'][0]
            
            left_pt_x, left_pt_y, right_pt_x, right_pt_y, closest_left_dist, closest_right_dist = utils.get_closest_boundary_pts(env, ego_scan, ego_x, ego_y, ego_theta)
            front_closest_pt_x, front_closest_pt_y, front_closest_dist = utils.get_front_closest_pt(ego_scan, ego_x, ego_y, ego_theta)
            
            # Test for point alignment checking
            left_oppo_hit_check = utils.check_target_pt_hit_oppo(env, left_pt_x, left_pt_y)
            right_oppo_hit_check = utils.check_target_pt_hit_oppo(env, right_pt_x, right_pt_y)
            front_oppo_hit_check = utils.check_target_pt_hit_oppo(env, front_closest_pt_x, front_closest_pt_y)

            if left_oppo_hit_check:
                print('Left point hit opponent')
            
            if right_oppo_hit_check:
                print('Right point hit opponent')
            
            if front_oppo_hit_check:
                print('Front point hit opponent')


            # Render closest pts
            vl1 = utils.render_line(env, left_pt_x, left_pt_y, ego_x, ego_y, color=(255, 0, 0))
            vl2 = utils.render_line(env, right_pt_x, right_pt_y, ego_x, ego_y, color=(0, 255, 0))
            vl3 = utils.render_line(env, front_closest_pt_x, front_closest_pt_y, ego_x, ego_y, color=(255, 255, 0))

            obs, timestep, done, _ = env.step(action)
            # print(action)
            if np.any(obs['collisions']):
                done = True
            laptime += timestep
            tracker_count += 1
            if render_sim:
                env.render('debug', debug_scale_factor=0.05)
            
            # Remove vertices list
            vl1.delete()
            vl2.delete()
            vl3.delete()
    print("finish one episode")