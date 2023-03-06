# MIT License

# Copyright (c) Hongrui Zheng, Johannes Betz

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Lattice Planner

Author: Hongrui Zheng
Last Modified: 6/13/22
"""
import logging
import os
from PIL import Image
import yaml
from argparse import Namespace
from scipy.ndimage import distance_transform_edt as edt
from es.utils.utils import *
from es.planner.pure_pursuit import AdvancedPurePursuitPlanner

from pyclothoids import Clothoid
import numpy as np
from numba import njit

logger = logging.getLogger(__name__)


class LatticePlanner:
    """

    """

    def __init__(self, conf, wb=0.33):
        """

        """
        self.wheelbase = wb

        ### load waypoints
        self.map_path = conf.map_path
        self.map_ext = conf.map_ext
        # waypoints (x, y, v, heading, s)
        waypoints = np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)
        waypoints = np.vstack((waypoints[:, 1], waypoints[:, 2], waypoints[:, 5], waypoints[:, 3], waypoints[:, 0])).T
        self.waypoints = waypoints
        self.traj_num = conf.traj_num
        self.lh_grid_lb = conf.lh_grid_lb
        self.lh_grid_ub = conf.lh_grid_ub
        self.lh_grid_rows = conf.lh_grid_rows
        self.traj_points = conf.traj_points
        self.traj_v_scale = conf.traj_v_scale
        self.s_max = self.waypoints[-1, 4]

        # sample and cost function
        self.sample_func = None
        self.add_sample_function(sample_lookahead_square)
        self.shape_cost_funcs = []
        self.constant_cost_funcs = []
        self.selection_func = None
        if conf.get_length_cost:
            self.add_shape_cost_function(get_length_cost)
        if conf.get_similarity_cost:
            self.add_shape_cost_function(get_similarity_cost)
        if conf.get_follow_optim_cost:
            self.add_shape_cost_function(get_follow_optim_cost)
        self.add_constant_cost_function(get_map_collision)

        self.params_num = conf.params_num
        self.params_name = conf.params_name
        self.params_idx = {}
        count = 0
        for name, num in zip(self.params_name, self.params_num):
            self.params_idx[name] = count
            count += num
        try:
            self.cost_weights_num = conf.weights_num
        except:
            self.cost_weights_num = 7
        self.set_cost_weights(self.cost_weights_num)

        self.v_lattice_span = np.linspace(conf.traj_v_span_min, conf.traj_v_span_max, conf.traj_v_span_num)
        self.v_lattice_num = conf.traj_v_span_num
        self.best_traj = None
        self.best_traj_ref_v = 0.0
        self.best_traj_idx = 0
        self.prev_traj_local = np.zeros((self.traj_points, 2))
        self.prev_opp_pose = np.array([0, 0])
        self.goal_grid = None
        self.state_i = None
        self.state_t = None
        self.step_all_cost = {}
        self.all_costs = None
        self.time_interval = conf.tracker_steps * 0.01
        self.last_s = 0.0

        #
        pure_pursuit_conf_file = conf.tracker_config_path
        with open(pure_pursuit_conf_file) as file:
            pp_conf = yaml.load(file, Loader=yaml.FullLoader)
        pp_conf = Namespace(**pp_conf)
        pp_conf.wpt_path = conf.wpt_path
        pp_conf.map_path = conf.map_path
        if conf.tracker == 'advanced_pure_pursuit':
            self.tracker = AdvancedPurePursuitPlanner(pp_conf)
        self.step = 0

        # load map image
        map_img_path = os.path.splitext(self.map_path)[0] + self.map_ext
        self.map_img = np.array(Image.open(map_img_path).transpose(Image.FLIP_TOP_BOTTOM)).astype(np.float64)
        self.map_img[self.map_img <= 128.] = 0.
        self.map_img[self.map_img > 128.] = 255.
        self.map_height = self.map_img.shape[0]
        self.map_width = self.map_img.shape[1]
        # load map yaml
        with open(self.map_path + '.yaml', 'r') as yaml_stream:
            map_metadata = yaml.safe_load(yaml_stream)
            self.map_resolution = map_metadata['resolution']
            self.origin = map_metadata['origin']

        self.orig_x = self.origin[0]
        self.orig_y = self.origin[1]
        self.orig_s = np.sin(self.origin[2])
        self.orig_c = np.cos(self.origin[2])

        self.dt = self.map_resolution * edt(self.map_img)
        self.map_metainfo = (
            self.orig_x, self.orig_y, self.orig_c, self.orig_s, self.map_height, self.map_width, self.map_resolution)

        # self.cost_weights = None

        # scan
        self.scan_num = conf.scan_num
        self.angle_span = np.linspace(-0.75 * np.pi, 0.75 * np.pi, self.scan_num)
        self.ittc_thres = conf.ittc_thres
        self.collision_thres = 0.3

    def add_shape_cost_function(self, func):
        """
        Add cost function to list for eval.
        """
        if type(func) is list:
            self.shape_cost_funcs.extend(func)
        else:
            self.shape_cost_funcs.append(func)

    def add_constant_cost_function(self, func):
        self.constant_cost_funcs.append(func)

    def set_parameters(self, parameters, v_scale=6.0):
        if type(parameters) == np.ndarray:
            for name, num in zip(self.params_name, self.params_num):
                start = self.params_idx[name]
                if name != 'cost_weights':
                    self.__setattr__(name, parameters[start])
                else:
                    self.set_cost_weights(parameters[start:start + num])
        else:
            for key, value in parameters.items():
                if key == 'cost_weights':
                    self.set_cost_weights(value)
                else:
                    self.__setattr__(key, value)

    def set_cost_weights(self, cost_weights):
        if type(cost_weights) == int:
            n = cost_weights
            cost_weights = np.array([1 / n] * n)
        elif len(cost_weights) != self.cost_weights_num:
            raise ValueError('Length of cost weights must be the same as number of cost functions.')
        self.cost_weights = cost_weights

    def add_sample_function(self, func):
        self.sample_func = func

    def add_selection_function(self, func):
        self.selection_func = func

    def sample(self, pose_x, pose_y, pose_theta, velocity, waypoints, lh_grid):
        """
        Sample a goal grid based on sample function. Given the current vehicle state, return a list of [x, y, theta] tuples

        Returns:
            goal_grid (numpy.ndarray [N, 3]): list of goal states, columns are [x, y, theta]
        """
        if self.sample_func is None:
            raise NotImplementedError('Please set a sample function before sampling.')

        goal_grid = self.sample_func(pose_x, pose_y, pose_theta, velocity, waypoints, lh_grid)

        return goal_grid

    def eval(self, all_traj, all_traj_clothoid, opp_poses, ego_pose):
        """
        Evaluate a list of generated clothoids based on added cost functions
        
        Args:
            pose_x
            pose_y
            pose_theta
        Returns:
            costs
        """
        cost_weights = self.cost_weights
        n, k = self.traj_num, self.v_lattice_num
        # cost = np.array([0.0] * self.traj_num)

        # assume use the **first** weight to calculate curvature cost
        mean_k, max_k = get_curvature(all_traj, all_traj_clothoid)  # (n, )
        cost = 2.0 * cost_weights[0] * mean_k
        self.step_all_cost['curvature_cost'] = cost.copy()

        ## other shape cost, current include length, similarity, map collision,
        for i, func in enumerate(self.shape_cost_funcs):
            cur_cost = func(all_traj, all_traj_clothoid, opp_poses, ego_pose, self.prev_traj_local, self.dt,
                            self.map_metainfo)
            cur_cost = cost_weights[i + 1] * cur_cost
            self.step_all_cost[func.__name__] = cur_cost
            cost += cur_cost

        ## cost functions with constant scale
        for i, func in enumerate(self.constant_cost_funcs):
            cur_cost = func(all_traj, all_traj_clothoid, opp_poses, ego_pose, self.prev_traj_local, self.dt,
                            self.map_metainfo, self.collision_thres)
            self.step_all_cost[func.__name__] = cur_cost
            cost += cur_cost

        ## velocity cost, assume use the last two weight to calculate abs velocity cost
        all_traj_min_mean_k = np.min(mean_k)
        mean_k_lattice = np.repeat(mean_k, k).reshape(n, k)  # (n, k)
        all_traj_v = all_traj[:, -1, 2]  # (n, )
        traj_v_lattice = np.repeat(all_traj_v, k).reshape(n, k) * self.v_lattice_span * self.traj_v_scale  # (n, k)
        # if 'curvature_speed_punish' in self.conf.params_dict.values:
        abs_v_cost = cost_weights[-3] * (np.max(traj_v_lattice) + 1 - traj_v_lattice) + \
                     cost_weights[-2] * (mean_k_lattice - all_traj_min_mean_k) * traj_v_lattice
        # abs_v_cost = cost_weights[-2] * (np.max(traj_v_lattice) + 1 - traj_v_lattice)
        # cost_weights[-2] * (mean_k_lattice - all_traj_min_mean_k) * traj_v_lattice
        self.step_all_cost['abs_v_cost'] = abs_v_cost

        ## collision cost
        collision_cost = cost_weights[-1] * get_obstacle_collision_with_v(all_traj, all_traj_clothoid, traj_v_lattice,
                                                                          opp_poses, self.prev_opp_pose,
                                                                          self.time_interval)
        self.step_all_cost['collision_cost'] = collision_cost

        cost = np.repeat(cost, k).reshape(n, k)
        # try:
        cost = cost + abs_v_cost + collision_cost
        # except:
        #     import ipdb; ipdb.set_trace()
        return cost

    def select(self, all_costs):
        """
        Select the best trajectory based on the selection function, defaults to argmin if no custom function is defined.
        """
        if self.selection_func is None:
            self.selection_func = np.argmin
        best_idx = self.selection_func(all_costs)
        return best_idx

    def init_rollout(self):
        self.prev_traj_local = np.zeros((self.traj_points, 2))
        self.prev_opp_pose = np.array([0, 0])
        self.last_s = 0.0
        self.step_all_cost = {}

    def cal_s(self):
        t, i = self.state_t, self.state_i
        s = self.waypoints[i, 4]
        if s < self.last_s:
            s = (s + self.s_max)
        self.last_s = s
        return s

    def generate_traj(self, ego_pose, velocity, waypoints=None, traj_points=None, lh_grid_lb=None, lh_grid_ub=None,
                      lh_grid_rows=None):
        if not traj_points:
            traj_points = self.traj_points
        if not lh_grid_lb:
            lh_grid_lb = self.lh_grid_lb
        if not lh_grid_ub:
            lh_grid_ub = self.lh_grid_ub
        if not lh_grid_rows:
            lh_grid_rows = self.lh_grid_rows
        if waypoints is None:
            waypoints = self.waypoints
        L = self.tracker.get_L(velocity)
        pose_x, pose_y, pose_theta = ego_pose
        lh_grid = np.linspace(L + lh_grid_lb, L + lh_grid_ub, lh_grid_rows)
        goal_grid = self.sample(pose_x, pose_y, pose_theta, velocity, waypoints, lh_grid)
        all_traj = []
        all_traj_clothoid = []
        for point in goal_grid:
            clothoid = Clothoid.G1Hermite(pose_x, pose_y, pose_theta, point[0], point[1], point[2])
            traj = sample_traj(clothoid, traj_points, point[3])
            all_traj.append(traj)
            # G1Hermite parameters are [xstart, ystart, thetastart, curvrate, kappastart, arclength]
            all_traj_clothoid.append(np.array(clothoid.Parameters))
        return np.array(all_traj), np.array(all_traj_clothoid), goal_grid

    def plan(self, pose_x, pose_y, pose_theta, opp_poses, velocity, waypoints=None):
        """
        Plan for next step

        Args:
            pose_x (float):
            pose_y (float):
            pose_theta (float):
            velocity (float):
            waypoints (numpy.ndarray [N, 5], optional, default=None):

        Returns:
            steering_angle (float):
            speed (float):
            selected_traj (numpy.ndarray [M, ])
        """
        # logger.debug(f'#STEP:{self.step}')
        self.step += 1
        if waypoints is None:
            waypoints = self.waypoints

        # state
        ego_pose = np.array([pose_x, pose_y, pose_theta])
        _, _, t, nearest_i = nearest_point(ego_pose[:2], waypoints[:, 0:2])
        self.state_i = nearest_i
        self.state_t = t
        # logger.debug(f'STATE:EGO_POSE:{ego_pose};OPPO_POSE:{opp_poses};EGO_V:{velocity}')

        # sample a grid based on current states
        min_L = self.tracker.get_L(velocity)
        lh_grid = np.linspace(min_L + self.lh_grid_lb, min_L + self.lh_grid_ub, self.lh_grid_rows)
        self.goal_grid = self.sample(pose_x, pose_y, pose_theta, velocity, waypoints, lh_grid)

        # generate clothoids
        all_traj = []
        all_traj_clothoid = []
        for point in self.goal_grid:
            clothoid = Clothoid.G1Hermite(pose_x, pose_y, pose_theta, point[0], point[1], point[2])
            traj = sample_traj(clothoid, self.traj_points, point[3])
            all_traj.append(traj)
            # G1Hermite parameters are [xstart, ystart, thetastart, curvrate, kappastart, arclength]
            all_traj_clothoid.append(np.array(clothoid.Parameters))
        # logger.debug(f'SAMPLE:ALL_TRAJ:{all_traj}')

        # evaluate all trajectory on all costs
        all_traj = np.array(all_traj)
        all_traj_clothoid = np.array(all_traj_clothoid)
        all_costs = self.eval(all_traj, all_traj_clothoid, opp_poses, ego_pose=ego_pose)
        self.all_costs = all_costs
        # all_costs_with_v = self.eval_with_v(all_traj, all_traj_clothoid, opp_poses, ego_pose=ego_pose)

        # select best trajectory

        best_traj_idx = self.select(all_costs)
        row_idx, col_idx = divmod(best_traj_idx, self.v_lattice_num)
        self.best_traj_idx = best_traj_idx
        # self.best_traj = all_traj[best_traj_idx]
        self.best_traj = all_traj[row_idx]
        self.best_traj_ref_v = self.best_traj[-1, 2]
        self.best_traj[:, 2] *= (self.v_lattice_span[col_idx] * self.traj_v_scale)
        self.prev_traj_local = traj_global2local(ego_pose, self.best_traj[:, :2])
        self.prev_opp_pose = opp_poses[:, :2]
        return self.best_traj

    def cal_objectives(self, rollout_states: dict, lap_time, overtake, crash):
        objectives = [0.0] * 2
        rollout_obs = rollout_states['rollout_obs']
        rollout_ego_s = rollout_states['rollout_ego_s']
        rollout_opp_s = rollout_states['rollout_opp_s']
        ego_control_error = np.sum(rollout_states['ego_control_error'])
        ego_ittc = np.array(rollout_states['ego_ittc'])
        abs_ittc = np.array(rollout_states['abs_ittc'])

        # progress
        ego_start_s, ego_end_s = rollout_ego_s[0], rollout_ego_s[-1]
        opp_start_s, opp_end_s = rollout_opp_s[0], rollout_opp_s[-1]
        ego_progress = ego_end_s - ego_start_s
        opp_progress = opp_end_s - opp_start_s
        relative_progress = opp_progress - ego_progress
        # objectives[0] = relative_progress * 5 / lap_time
        objectives[0] = opp_end_s - ego_end_s

        # safety
        # objectives[1] = 5 * (10 * ego_control_error / n - np.sum(ego_ittc) * 0.2 / n)
        # objectives[1] = len(np.nonzero(ego_ittc)) / n + len(np.nonzero(ego_ittc < self.ittc_thres)) / n

        n = len(ego_ittc)
        if n == 0:
            return [0, 0]
        # import ipdb;
        # ipdb.set_trace()

        # print(ego_control_error / n, np.sum(abs_ittc) / n)
        objectives[1] += len(np.nonzero(ego_ittc)[0]) / n
        ego_ittc = ego_ittc[ego_ittc > 0]
        objectives[1] += len(np.nonzero(ego_ittc < self.ittc_thres)[0]) / n
        ego_ittc = ego_ittc[ego_ittc < self.ittc_thres]
        objectives[1] *= self.ittc_thres
        if len(ego_ittc != 0):
            objectives[1] += (self.ittc_thres - np.mean(ego_ittc))

        # objectives[1] += 10 * (ego_control_error / n - np.sum(abs_ittc) / n)
        objectives[1] += 10 * (ego_control_error / n)
        #  print(np.sum(ego_ittc))

        if overtake:
            objectives[0] *= 1.1
        if crash:
            objectives[1] += 1

        return objectives

    def cal_batch_objective(self, objectives, batch_crash, batch_overtake):
        n = len(objectives)
        mean_obj = np.mean(objectives, axis=0)
        mean_obj[0] += 5.0 * (n - batch_overtake) / n
        mean_obj[1] += 10.0 * batch_crash / n
        # mean_obj[0] -= batch_overtake
        # mean_obj[1] += batch_crash
        return mean_obj


"""

General1 function for sampling a grid of goal points

"""


@njit(cache=True)
def sample_lookahead_square(pose_x,
                            pose_y,
                            pose_theta,
                            velocity,
                            waypoints,
                            lookahead_distances=np.array([1.6, 1.8, 2.0, 2.2]),
                            widths=np.linspace(-1.5, 1.5, num=11)):
    """
    General1 function to sample goal points. In this example it samples a rectangular grid around a look-ahead point.

    TODO: specify waypoints idx somehow? as planner arguments?

    Args:
        pose_x ():
        pose_y ():
        pose_theta ():
        velocity ():
        waypoints ():
        lookahead_distances ():
        widths ():
    
    Returns:
        grid (numpy.ndarray (n, 3)): Returned grid of goal points
    """
    # get lookahead points to create grid along waypoints
    position = np.array([pose_x, pose_y])
    nearest_p, nearest_dist, t, nearest_i = nearest_point(position, waypoints[:, 0:2])
    local_span = np.vstack((np.zeros_like(widths), widths))
    xy_grid = np.zeros((2, 1))
    theta_grid = np.zeros((len(lookahead_distances), 1))
    v_grid = np.zeros((len(lookahead_distances), 1))
    for i, d in enumerate(lookahead_distances):
        lh_pt, i2, t2 = intersect_point(np.ascontiguousarray(nearest_p), d, waypoints[:, 0:2], t + nearest_i, wrap=True)
        i2 = int(i2)
        # different heading
        lh_pt_theta = waypoints[i2, 3] + 0.5 * np.pi
        lh_pt_v = waypoints[i2, 2]
        lh_span_points = get_rotation_matrix(lh_pt_theta) @ local_span + lh_pt.reshape(2, -1)
        xy_grid = np.hstack((xy_grid, lh_span_points))
        theta_grid[i] = zero_2_2pi(lh_pt_theta)
        v_grid[i] = lh_pt_v
    xy_grid = xy_grid[:, 1:]
    theta_grid = np.repeat(theta_grid, len(widths)).reshape(1, -1)
    v_grid = np.repeat(v_grid, len(widths)).reshape(1, -1)
    grid = np.vstack((xy_grid, theta_grid, v_grid)).T
    return grid


"""

Tool function(maybe move to util.py afterwards)

"""


@njit(cache=True)
def traj_global2local(ego_pose, traj):
    """
    traj: (n, m, 2) or (m, 2)
    """
    new_traj = np.zeros_like(traj)
    pose_x, pose_y, pose_theta = ego_pose
    c = np.cos(pose_theta)
    s = np.sin(pose_theta)
    new_traj[..., 0] = c * (traj[..., 0] - pose_x) + s * (traj[..., 1] - pose_y)  # (n, m, 1)
    new_traj[..., 1] = -s * (traj[..., 0] - pose_x) + c * (traj[..., 1] - pose_y)  # (n, m, 1)
    return new_traj


"""

General1 functions for different costs

traj: np.ndarray, (n, m, 5), (x, y, v, heading, arc_length)
traj_clothoid: np.ndarray, (n, 6), (x0, y0, theta0, k0, dk, arc_length)
opp_poses: np.ndarray, (k, 3)

Return: 
cost: np.ndarray, (n, 1) 
"""


@njit(cache=True)
def get_follow_optim_cost(traj, traj_clothoid, opp_poses=None, ego_pose=None, prev_traj=None, dt=None,
                          map_metainfo=None):
    scale = 0.1
    n = traj.shape[0]
    center = np.array((5, 16, 27))
    center = np.repeat(center, 11)  # (33, 0)
    traj_idx = np.arange(0, n, 1)
    idx_diff = traj_idx - center
    cost = idx_diff * idx_diff
    return cost * scale


def get_curvature(traj, traj_clothoid):
    k0 = traj_clothoid[:, 3].reshape(-1, 1)  # (n, 1)
    dk = traj_clothoid[:, 4].reshape(-1, 1)  # (n, 1)
    s = traj_clothoid[:, -1]  # (n, )
    s_pts = np.linspace(np.zeros_like(s), s, num=traj.shape[1]).T  # (n, m)
    traj_k = k0 + dk * s_pts  # (n, m)
    traj_k = np.abs(traj_k)
    max_k = np.max(traj_k, axis=1)
    mean_k = np.mean(traj_k, axis=1)
    return mean_k, max_k


@njit(cache=True)
def get_length_cost(traj, traj_clothoid, opp_poses=None, ego_pose=None, prev_traj=None, dt=None, map_metainfo=None):
    scale = 0.5
    return traj_clothoid[:, -1] * scale


def get_similarity_cost(traj, traj_clothoid, opp_poses=None, ego_pose=None, prev_traj=None, dt=None, map_metainfo=None):
    """
    the stored prev_traj is in local frame
    """
    scale = 0.25
    if prev_traj[-1][-1] == 0:
        return np.zeros((len(traj)))
    # prev_traj = prev_traj[:, :2]
    local_traj = traj_global2local(ego_pose, traj[..., :2])
    diff = local_traj - prev_traj
    cost = diff * diff
    cost = np.sum(cost, axis=(1, 2)) * scale
    return cost


@njit(cache=True)
def get_map_collision(traj, traj_clothoid, opp_poses=None, ego_pose=None, prev_traj=None, dt=None, map_metainfo=None,
                      collision_thres=0.4):
    if dt is None:
        raise ValueError('Map Distance Transform dt has to be set when using this cost function.')
    # points: (n, 2)
    all_traj_pts = np.ascontiguousarray(traj).reshape(-1, 5)  # (nxm, 5)
    collisions = map_collision(all_traj_pts[:, 0:2], dt, map_metainfo, eps=collision_thres)  # (nxm)
    collisions = collisions.reshape(len(traj), -1)  # (n, m)
    cost = []
    for traj_collision in collisions:
        if np.any(traj_collision):
            cost.append(10.0)
        else:
            cost.append(0.)
    return np.array(cost)


@njit(cache=True)
def get_obstacle_collision_with_v(traj, traj_clothoid, v_lattice, opp_poses, prev_oppo_pose, dt=None):
    max_cost = 10.0
    min_cost = 10.0
    width, length = 0.31, 0.58
    n, m, _ = traj.shape
    k = v_lattice.shape[1]
    cost = np.zeros(n)
    traj_xyt = traj[:, :, :3]
    for i, tr in enumerate(traj_xyt):
        # print(i)
        close_p_idx = x2y_distances_argmin(np.ascontiguousarray(opp_poses[:, :2]), np.ascontiguousarray(tr[:, :2]))
        for opp_pose, p_idx in zip(opp_poses, close_p_idx):
            opp_box = get_vertices(opp_pose, length, width)
            p_box = get_vertices(tr[int(p_idx)], length, width)
            if collision(opp_box, p_box):
                cost[i] = max_cost - p_idx * (max_cost - min_cost) / m
    if np.sum(prev_oppo_pose) == 0:
        cost = np.repeat(cost, k).reshape(n, k)
        return cost
    else:
        cost = np.repeat(cost, k).reshape(n, k)
        # calculate opp pose, assume only one opponent
        oppo_pose = opp_poses[0][:2]
        prev_opp_pose = prev_oppo_pose[0]
        opp_v = (oppo_pose - prev_opp_pose) / float(dt)  # (2, 1)
        # print(opp_v)
        traj_heading = traj[:, -1, 3]  # (n, )
        traj_heading_vec = np.vstack((np.cos(traj_heading), np.sin(traj_heading))).T  # (n, 2)
        opp_v_proj = np.dot(traj_heading_vec, opp_v.reshape(2, 1))  # (n, 1)
        opp_v_proj = np.repeat(opp_v_proj, k).reshape(n, k)  # (n, k)
        v_diff = v_lattice - opp_v_proj
        cost = cost * v_diff
        for i in range(n):
            for j in range(k):
                if cost[i][j] < 0:
                    cost[i][j] = 1.0
                elif cost[i][j] < 1.2:
                    cost[i][j] = 1.2
    return cost
