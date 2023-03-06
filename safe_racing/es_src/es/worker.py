# MIT License

# Copyright (c) 2022 Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Remote worker of gradient-free optimization. Runs evaluation.

Author: Hongrui Zheng
Last Modified: 6/27/2022
"""
import numpy as np
import ray
import gym

# TODO: import planner
from es.planner.lattice_planner import LatticePlanner
from es.utils.utils import *
from es.utils.DataProcessor import obsDict2carStateSeq, paramsDict2Array
from es.utils.utils import S_Plus

import pickle
import time
import os
from es.utils.visualize import LatticePlannerRender
from es.utils.DataProcessor import RolloutDataLogger


@ray.remote
class Worker:
    """
    Ray remote sim worker
    """

    def __init__(self, conf, lp_conf, worker_id, run_id):
        # score
        self.objetives = [0.0] * 2
        self.eval_done = False

        self.conf = conf
        self.worker_id = worker_id
        self.run_id = run_id
        self.batch_file_prefix = os.path.join(os.path.abspath(f'../runs/{run_id}/batch_data'), f'worker_{worker_id}')
        self.run_file_prefix = os.path.abspath(f'../runs/{run_id}')

        # load waypoints for generate scenarios
        self.rng = np.random.default_rng(self.conf.seed)
        self.ego_planner = LatticePlanner(lp_conf)
        self.opp_planner = LatticePlanner(lp_conf)
        self.map_path = lp_conf.map_path
        self.wpt_xyhs = np.vstack(
            (self.ego_planner.waypoints[:, 0], self.ego_planner.waypoints[:, 1],
             self.ego_planner.waypoints[:, 3], self.ego_planner.waypoints[:, 4])).T
        self.wpt_num = len(self.wpt_xyhs)
        self.waypoints_ptnum = len(self.wpt_xyhs)
        self.num_agents = 2

        self.logger = RolloutDataLogger(log_location=self.batch_file_prefix, rollout_states=conf.rollout_states)
        self.simenv = gym.make('f110_gym:f110-v0', map=self.map_path, map_ext='.png', num_agents=self.num_agents)
        if conf.render:
            self.render = LatticePlannerRender(self.ego_planner)
            self.simenv.add_render_callback(self.render.render_callback)
        self.tracker_steps = lp_conf.tracker_steps
        # scenarios
        self.ego_start_pose = None
        self.opp_start_pose = None
        self.scene_num = self.conf.num_scene
        self.opp_num = self.conf.num_opp
        self.opp_gap_min = self.conf.opp_gap_min
        self.opp_gap_max = self.conf.opp_gap_max
        self.xy_noise_s = self.conf.xy_noise_scale
        self.theta_noise_s = self.conf.theta_noise_scale

        # store data
        self.batch_ego_s = []
        self.batch_opp_s = []
        self.batch_ego_weights = []
        self.batch_opp_weights = []
        self.batch_obs = []
        self.batch_done = []
        self.batch_best_traj = []
        self.batch_grid_goal = []
        self.batch_error = []
        self.batch_crash = 0
        self.batch_overtake = 0

        self.generate_scenarios_opp()
        # import ipdb; ipdb.set_trace()

    def generate_scenarios_opp(self):
        opp_num = self.opp_num
        opp_gap_min, opp_gap_max = self.opp_gap_min, self.opp_gap_max
        xy_noise_s, theta_noise_s = self.conf.xy_noise_scale, self.conf.theta_noise_scale

        idx_for_shuffle = np.arange(0, self.waypoints_ptnum - self.opp_gap_max, 3)
        ego_seg_idx = self.rng.choice(idx_for_shuffle, opp_num, replace=False)  # (seg_num, )
        ego_start_pose = self.wpt_xyhs[ego_seg_idx][:, :3]  # (n, 3)

        oppo_gap = np.arange(opp_gap_min, opp_gap_max, 1)
        oppo_seg_idx = self.rng.choice(oppo_gap, opp_num, replace=True)
        oppo_seg_idx = ego_seg_idx + oppo_seg_idx
        oppo_start_pose = self.wpt_xyhs[oppo_seg_idx][:, :3]

        ego_start_pose[:, 2] = (ego_start_pose[:, 2] + 2.5 * np.pi) % (2 * np.pi)
        oppo_start_pose[:, 2] = (oppo_start_pose[:, 2] + 2.5 * np.pi) % (2 * np.pi)

        # add noise
        n = len(ego_start_pose)
        ego_start_pose = ego_start_pose + \
                         np.hstack((self.rng.random(size=(n, 2)) * xy_noise_s,
                                    self.rng.random(size=(n, 1)) * theta_noise_s))
        oppo_start_pose = oppo_start_pose + \
                          np.hstack((self.rng.random(size=(n, 2)) * xy_noise_s,
                                     self.rng.random(size=(n, 1)) * theta_noise_s))

        self.ego_start_pose = ego_start_pose
        self.opp_start_pose = oppo_start_pose

        if self.worker_id == 0:
            npz_file = os.path.join(self.run_file_prefix, 'rollout_scenarios.npz')
            np.savez_compressed(npz_file, ego_start_pose=ego_start_pose, opp_start_pose=oppo_start_pose)

    def generate_scenarios_oppscene(self):
        scene_num, opp_num = self.scene_num, self.opp_num
        opp_gap_min, opp_gap_max = self.opp_gap_min, self.opp_gap_max
        xy_noise_s, theta_noise_s = self.conf.xy_noise_scale, self.conf.theta_noise_scale

        # current no end track
        idx_for_shuffle = np.arange(0, self.waypoints_ptnum - self.opp_gap_max, 5)
        ego_seg_idx = self.rng.choice(idx_for_shuffle, scene_num, replace=False)  # (seg_num, )
        ego_seg_idx = np.repeat(ego_seg_idx, opp_num)  # (1, 1, 1, 2, 2, 2), seg_num*oppo_num, like (row, col).flatten
        ego_start_pose = self.wpt_xyhs[ego_seg_idx][:, :3]  # (n, 3)

        oppo_gap = np.arange(opp_gap_min, opp_gap_max, 1)
        oppo_seg_idx = self.rng.choice(oppo_gap, opp_num, replace=True)
        # oppo_seg_idx[0] = -oppo_seg_idx[0]

        oppo_seg_idx = np.repeat(oppo_seg_idx.reshape(-1, 1), scene_num,
                                 axis=1).T.flatten()  # (oppo_num, seg_num).T.flatten(), (seg_num*oppo_num)
        oppo_seg_idx = ego_seg_idx + oppo_seg_idx
        oppo_start_pose = self.wpt_xyhs[oppo_seg_idx][:, :3]

        # import ipdb; ipdb.set_trace()
        # ego_start_pose[:, 2][np.nonzero(ego_start_pose[:, 2] < 0)[0]] += 2*np.pi
        # oppo_start_pose[:, 2][np.nonzero(oppo_start_pose[:, 2] < 0)[0]] += 2 * np.pi
        ego_start_pose[:, 2] = (ego_start_pose[:, 2] + 2.5 * np.pi) % (2 * np.pi)
        oppo_start_pose[:, 2] = (oppo_start_pose[:, 2] + 2.5 * np.pi) % (2 * np.pi)

        # add noise
        n = len(ego_start_pose)
        ego_start_pose = ego_start_pose + \
                         np.hstack((self.rng.random(size=(n, 2)) * xy_noise_s,
                                    self.rng.random(size=(n, 1)) * theta_noise_s))
        oppo_start_pose = oppo_start_pose + \
                          np.hstack((self.rng.random(size=(n, 2)) * xy_noise_s,
                                     self.rng.random(size=(n, 1)) * theta_noise_s))

        self.ego_start_pose = ego_start_pose.reshape(self.scene_num, self.opp_num, 3)
        self.opp_start_pose = oppo_start_pose.reshape(self.scene_num, self.opp_num, 3)

        # Tested, same pose in every worker, same shape(s_num, opp_num, 3) works
        # print(self.ego_start_pose)
        # print(self.opp_start_pose)

    def run_sim(self, ego_param, opponent_weights, epoch, scale=1.0, time_per_run=8.0):
        """
        Run simulation with given work

        Args:
            ego_param (dict): genome to be evaluated
            opponent_weights (np.ndarray (nopp, nweights)):
        Returns:
            None
        """
        os.makedirs(os.path.join(self.batch_file_prefix, f'epoch_{epoch}'), exist_ok=True)

        ## deal with parameters
        ego_weights_np = paramsDict2Array(ego_param, self.ego_planner.params_name)
        # for i in range(len(opponent_weights)):
        #     opp_params.append({'traj_v_scale': opponent_weights[i][0], 'cost_weights': opponent_weights[i][1:-1],
        #                        'collision_thres': opponent_weights[i][-1]})
        if self.conf.render:
            print('================ ego parameters =================')
            print(ego_param)
        # self.ego_planner.set_cost_weights(ego_weights)
        self.ego_planner.set_parameters(ego_param)
        # print(ego_weights)
        batch_objective = []
        batch_overtake = 0
        batch_crash = 0
        s_idx = 0
        for o_idx in range(self.opp_num):
            self.opp_planner.set_parameters(opponent_weights[o_idx])
            self.ego_planner.init_rollout()
            self.opp_planner.init_rollout()
            self.logger.clear_buffer()
            agents_pose = np.vstack((self.ego_start_pose[o_idx], self.opp_start_pose[o_idx]))
            obs, _, done, _ = self.simenv.reset(agents_pose)
            if self.conf.render:
                self.simenv.render()

            rollout_step = 0
            laptime = 0
            ego_s = 0.0
            opp_s = 0.0
            overtake = False
            crash = False
            _, last_dist = cal_single_ittc(pose_1=agents_pose[0].flatten(),
                                           pose_2=agents_pose[1].flatten(), dt=0.01 * self.tracker_steps)
            step_states = {
                'rollout_obs': obs
            }
            self.logger.update_buffer(step_states)

            while not done and laptime < time_per_run:
                rollout_step += 1
                oppo_pose = obsDict2oppoArray(obs, 0)
                ego_best_traj = self.ego_planner.plan(obs['poses_x'][0], obs['poses_y'][0],
                                                      obs['poses_theta'][0], oppo_pose,
                                                      obs['linear_vels_x'][0])
                oppo_pose = obsDict2oppoArray(obs, 1)
                opp_best_traj = self.opp_planner.plan(obs['poses_x'][1], obs['poses_y'][1], obs['poses_theta'][1],
                                                      oppo_pose,
                                                      obs['linear_vels_x'][1])

                tracker_count = 0

                while not done and tracker_count < self.tracker_steps:
                    ego_steer, ego_speed = self.ego_planner.tracker.plan(obs['poses_x'][0], obs['poses_y'][0],
                                                                         obs['poses_theta'][0],
                                                                         obs['linear_vels_x'][0], ego_best_traj)
                    opp_steer, opp_speed = self.opp_planner.tracker.plan(obs['poses_x'][1], obs['poses_y'][1],
                                                                         obs['poses_theta'][1],
                                                                         obs['linear_vels_x'][1], opp_best_traj)
                    action = np.array([[ego_steer, ego_speed], [opp_steer, opp_speed]])
                    obs, timestep, done, _ = self.simenv.step(action)
                    if np.any(obs['collisions']):
                        done = True
                    laptime += timestep
                    tracker_count += 1
                    if self.conf.render:
                        self.simenv.render()

                    if tracker_count != 0:
                        step_states = {
                            'ego_control_error': self.ego_planner.tracker.nearest_dist
                        }
                        self.logger.update_buffer(step_states)

                ################ Record Data for each timestep ###############

                ################ Record Data for each timestep ###############
                ego_s = self.ego_planner.cal_s()
                opp_s = self.opp_planner.cal_s()

                if ego_s > opp_s and rollout_step < 2:
                    opp_s = opp_s + self.ego_planner.s_max
                    self.opp_planner.last_s = opp_s

                # abs ittc
                ego_distance = obs['scans'][0]  # (scan_num, 1)
                ego_speed_proj = np.cos(self.ego_planner.angle_span) * obs['linear_vels_x'][0]  # (scan_num, 1)
                ego_speed_proj[ego_speed_proj <= 0.0] = 0.001
                raw_ittc = ego_distance / ego_speed_proj
                # abs_ittc = np.min(raw_ittc)
                if np.min(raw_ittc) > self.ego_planner.ittc_thres:
                    abs_ittc = 0.0
                else:
                    abs_ittc = np.min(raw_ittc)


                # ittc with oppo
                if ego_s < opp_s:
                    ittc, last_dist = cal_single_ittc(
                        pose_1=np.array((obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0])),
                        pose_2=np.array((obs['poses_x'][1], obs['poses_y'][1], obs['poses_theta'][1])),
                        last_dist=last_dist, dt=0.01 * self.tracker_steps
                    )
                    if ittc <= 0.0:
                        ittc = 0.0
                else:
                    ittc = np.nan
                # import time
                # begin = time.time()
                # update buffer
                step_states = {
                    'rollout_obs': obs,
                    'ego_best_traj_idx': self.ego_planner.best_traj_idx,
                    'opp_best_traj_idx': self.opp_planner.best_traj_idx,
                    'ego_prev_traj': self.ego_planner.prev_traj_local,
                    'ego_ittc': ittc,
                    'rollout_ego_s': ego_s,
                    'rollout_opp_s': opp_s,
                    'abs_ittc': abs_ittc
                }
                self.logger.update_buffer(step_states)

            if rollout_step < 2:
                print(f'invalid rollout, crash at first')
                continue
            if ego_s > opp_s:
                overtake = True
                batch_overtake += 1
            if obs['collisions'][0] and not overtake:
                crash = True
                batch_crash += 1

            self.logger.remove_nan('ego_ittc')
            rollout_objectives = self.ego_planner.cal_objectives(self.logger.rollout_buffer, laptime, overtake, crash)
            rollout_objectives = np.clip(rollout_objectives, -50, 50)
            if rollout_objectives[0] == 0 and rollout_objectives[1]==0:
                print(f'invalid rollout, crash at first or overflow')
                continue
            ################ Save data for each rollout #################
            features = {
                'ego_weights': ego_param,
                'opp_weights': opponent_weights[o_idx],
                'objectives': rollout_objectives
            }
            # self.logger.save_rollout_data(epoch=epoch, o_idx=o_idx, s_idx=0, **features)
            ################ Save data for each rollout #################
            batch_objective.append(rollout_objectives)
            if self.conf.render:
                print(
                    f'scene_id:{s_idx}, oppo_id:{o_idx}, laptime:{np.round(laptime, 2)}, objectives:{np.round(rollout_objectives, 2)},')

            #     f'ego_weights: {self.ego_planner.cost_weights}, opp_weights: {self.opp_planner.cost_weights}')
        batch_objective = np.array(batch_objective)  # (n, 3)
        # import ipdb; ipdb.set_trace()
        self.batch_crash = batch_crash
        self.batch_overtake = batch_overtake
        self.objetives = self.cal_batch_objective(batch_objective, batch_crash, batch_overtake)
        self.eval_done = True
        print(
            f'finish one batch, batch objectives are {self.objetives}, overtake is {batch_overtake}, crash is {batch_crash}')

    def cal_batch_objective(self, objectives, batch_crash, batch_overtake):
        n = len(objectives)
        mean_obj = np.mean(objectives, axis=0)
        mean_obj[0] += 5.0 * (n - batch_overtake) / n
        mean_obj[1] += 10.0 * batch_crash / n
        # mean_obj[0] -= batch_overtake
        # mean_obj[1] += batch_crash
        return mean_obj

    def collect(self):
        """
        Collect function, called when eval result is requested
        Resets worker instance after called

        Args:
            None

        Returns:
            objectives ([float]): objective scores of the current evaluation
        """
        while not self.eval_done:
            continue
        # return self.objetives, self.batch_obs, self.batch_grid_goal, self.batch_best_traj, self.batch_ego_weights, self.batch_opp_weights
        return self.objetives, self.batch_crash, self.batch_overtake


def calculate_objectives(rollout_ego_s, rollout_opp_s, rollout_obs, rollout_done, rollout_error, s_max, lap_time,
                         rollout_ittc, overtake, crash):
    # TODO: normalization
    # 10.0
    objectives = [0.0] * 2
    ego_states = obsDict2carStateSeq(rollout_obs, ego_idx=0)
    opp_states = obsDict2carStateSeq(rollout_obs, ego_idx=1)
    n = len(ego_states['pose_x'])
    if n < 3:
        return objectives

    # progress
    ego_start_s, ego_end_s = rollout_ego_s[0], rollout_ego_s[-1]
    opp_start_s, opp_end_s = rollout_opp_s[0], rollout_opp_s[-1]
    ego_progress = ego_end_s - ego_start_s
    opp_progress = opp_end_s - opp_start_s
    relative_progress = opp_progress - ego_progress
    objectives[0] = relative_progress * 10 / lap_time

    # ittc
    objectives[1] = 5 * (10 * rollout_error / n - np.sum(rollout_ittc) / n)

    if overtake:
        objectives[0] *= 1.1
        # objectives[1] *= 0.5
    if crash:
        objectives[0] *= 1.1

    return objectives

# ego_s = self.wpt_xyhs[ego_i, 3] + \
#         ego_t * S_Plus(self.wpt_xyhs[S_Plus(ego_i, 1, self.wpt_num), 3],
#                        -self.wpt_xyhs[ego_i, 3],
#                        self.s_max)
