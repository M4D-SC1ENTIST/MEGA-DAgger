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
Pure Pursuit waypoint tracker

Author: Hongrui Zheng
Last Modified: 5/4/22
"""

from es.utils.utils import *

try:
    from pyglet.gl import GL_POINTS
except:
    pass

import numpy as np
import warnings


class PurePursuitPlanner():
    """
    Pure pursuit tracking controller
    Reference: Coulter 1992, https://www.ri.cmu.edu/pub_files/pub3/coulter_r_craig_1992_1/coulter_r_craig_1992_1.pdf

    All vehicle pose used by the planner should be in the map frame.

    Args:
        waypoints (numpy.ndarray [N x 4], optional): static waypoints to track

    Attributes:
        max_reacquire (float): maximum radius (meters) for reacquiring current waypoints
        waypoints (numpy.ndarray [N x 4]): static list of waypoints, columns are [x, y, velocity, heading]
    """

    def __init__(self, wheelbase=0.33, waypoints=None):
        self.max_reacquire = 20.
        self.wheelbase = wheelbase
        self.waypoints = waypoints

    def _get_current_waypoint(self, lookahead_distance, position, theta):
        """
        Finds the current waypoint on the look ahead circle intersection

        Args:
            lookahead_distance (float): lookahead distance to find next point to track
            position (numpy.ndarray (2, )): current position of the vehicle (x, y)
            theta (float): current vehicle heading

        Returns:
            current_waypoint (numpy.ndarray (3, )): selected waypoint (x, y, velocity), None if no point is found
        """

        nearest_p, nearest_dist, t, i = nearest_point(position, self.waypoints[:, 0:2])

        if nearest_dist < lookahead_distance:
            lookahead_point, i2, t2 = intersect_point(position,
                                                      lookahead_distance,
                                                      self.waypoints[:, 0:2],
                                                      i + t,
                                                      wrap=True)
            if i2 is None:
                all_distance = np.linalg.norm(self.waypoints[:, 0:2] - position, axis=1)
                all_distance_lh = np.abs(all_distance - lookahead_distance)
                best_p_idx = np.argmin(all_distance_lh)
                return self.waypoints[best_p_idx, :]
            current_waypoint = np.array([self.waypoints[i2, 0], self.waypoints[i2, 1], self.waypoints[i, 2]])
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            return self.waypoints[i, :]
        else:
            return None

    def plan(self, pose_x, pose_y, pose_theta, lookahead_distance, waypoints=None):
        """
        Planner plan function overload for Pure Pursuit, returns acutation based on current state

        Args:
            pose_x (float): current vehicle x position
            pose_y (float): current vehicle y position
            pose_theta (float): current vehicle heading angle
            lookahead_distance (float): lookahead distance to find next waypoint to track
            waypoints (numpy.ndarray [N x 4], optional): list of dynamic waypoints to track, columns are [x, y, velocity, heading]

        Returns:
            speed (float): commanded vehicle longitudinal velocity
            steering_angle (float):  commanded vehicle steering angle
        """
        if waypoints is not None:
            if waypoints.shape[1] < 3 or len(waypoints.shape) != 2:
                raise ValueError('Waypoints needs to be a (Nxm), m >= 3, numpy array!')
            self.waypoints = waypoints
        else:
            if self.waypoints is None:
                raise ValueError('Please set waypoints to track during planner instantiation or when calling plan()')
        position = np.array([pose_x, pose_y])
        lookahead_point = self._get_current_waypoint(lookahead_distance,
                                                     position,
                                                     pose_theta)
        new_L = np.linalg.norm(lookahead_point[:2] - position)
        # if lookahead_point is None:
        #     warnings.warn('Cannot find lookahead point, stopping...')
        #     return 0.0, 0.0

        speed, steering_angle = get_actuation(pose_theta,
                                              lookahead_point,
                                              position,
                                              new_L,
                                              self.wheelbase)

        return steering_angle, speed


class AdvancedPurePursuitPlanner:
    def __init__(self, conf, wb=0.33):
        """
        conf: NameSpace
        """

        self.wheelbase = wb
        self.conf = conf
        self.max_reacquire = 20.

        self.drawn_waypoints = []
        # waypoint index
        self.wpt_xind = self.conf.wpt_xind
        self.wpt_yind = self.conf.wpt_yind
        self.wpt_vind = self.conf.wpt_vind
        self.waypoints_xyv = None
        self.waypoints = None
        self.load_waypoints(conf)
        self.wpNum = self.waypoints.shape[0]

        # advanced pure pursuit
        self.minL = self.conf.minL
        self.maxL = self.conf.maxL
        self.minP = self.conf.minP
        self.maxP = self.conf.maxP
        self.Pscale = self.conf.Pscale
        self.Lscale = self.conf.Lscale
        self.D = self.conf.D
        self.vel_scale = self.conf.vel_scale
        self.prev_error = 0.0
        self.interpScale = self.conf.interpScale

        # ittc

        self.debug = self.conf.debug

    def _change_waypoint_xyv_idx(self, new_x_idx, new_y_idx, new_v_idx):
        self.wpt_xind = new_x_idx
        self.wpt_yind = new_y_idx
        self.wpt_vind = new_v_idx
        print('change waypoint x, y, v idx')

    def load_waypoints(self, conf):
        """
        loads waypoints
        """
        waypoints = np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)
        waypoints = np.vstack((waypoints[:, 1], waypoints[:, 2], waypoints[:, 5], waypoints[:, 3], waypoints[:, 0])).T
        self.waypoints = waypoints
        self.waypoints_xyv = waypoints[:, :3]

    def render_waypoints(self, e):
        """
        update waypoints being drawn by EnvRenderer
        """

        # points = self.waypoints

        points = np.vstack((self.waypoints[:, 0], self.waypoints[:, 1])).T

        scaled_points = 50. * points

        for i in range(points.shape[0]):
            if len(self.drawn_waypoints) < points.shape[0]:
                b = e.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_points[i, 0], scaled_points[i, 1], 0.]),
                                ('c3B/stream', [183, 193, 222]))
                self.drawn_waypoints.append(b)
            else:
                self.drawn_waypoints[i].vertices = [scaled_points[i, 0], scaled_points[i, 1], 0.]

    def _get_current_waypoint(self, lookahead_distance, position, theta, waypoints):
        """
        gets the current waypoint to follow
        """
        nearest_p, nearest_dist, t, i = nearest_point(position, self.waypoints[:, 0:2])
        # import ipdb;
        # ipdb.set_trace()
        if nearest_dist < lookahead_distance:
            lookahead_point, i2, t2 = intersect_point(position,
                                                      lookahead_distance,
                                                      self.waypoints[:, 0:2],
                                                      i + t,
                                                      wrap=True)
            if i2 is None:
                all_distance = np.linalg.norm(self.waypoints[:, 0:2] - position, axis=1)
                all_distance_lh = np.abs(all_distance - lookahead_distance)
                best_p_idx = np.argmin(all_distance_lh)
                return self.waypoints[best_p_idx, :]
            current_waypoint = np.array([self.waypoints[i2, 0], self.waypoints[i2, 1], self.waypoints[i, 2]])
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            return self.waypoints[i, :]
        else:
            return None

    def get_L(self, curr_v):
        return curr_v * (self.maxL - self.minL) / self.Lscale + self.minL

    def plan(self, pose_x, pose_y, pose_theta, curr_v, waypoints):

        # get L, P with speed
        L = curr_v * (self.maxL - self.minL) / self.Lscale + self.minL
        P = self.maxP - curr_v * (self.maxP - self.minP) / self.Pscale

        position = np.array([pose_x, pose_y])
        lookahead_point, new_L, nearest_dist = get_wp_xyv_with_interp(L, position, pose_theta, waypoints,
                                                                      waypoints.shape[0], self.interpScale)
        self.nearest_dist = nearest_dist

        speed, steering, error = \
            get_actuation_PD(pose_theta, lookahead_point, position, new_L, self.wheelbase, self.prev_error, P, self.D)
        speed = speed * self.vel_scale
        self.prev_error = error

        # if self.debug:
        # print(f'target_speed: {speed},  current_speed: {curr_v}, steering: {steering}')
        # print(f'L: {L},  P:{P}, error: {error}')
        # print(f'L:{L}, new_L:{new_L}')
        # if np.any(np.isnan(np.array([steering, speed]))):
        #     import ipdb
        #     ipdb.set_trace()
        return steering, speed


@njit(cache=True)
def simple_norm_axis1(vector):
    return np.sqrt(vector[:, 0] ** 2 + vector[:, 1] ** 2)


@njit(cache=True)
def get_wp_xyv_with_interp(L, curr_pos, theta, waypoints, wpNum, interpScale):
    traj_distances = simple_norm_axis1(waypoints[:, :2] - curr_pos)
    nearest_idx = np.argmin(traj_distances)
    nearest_dist = traj_distances[nearest_idx]
    segment_end = nearest_idx
    # count = 0
    if wpNum < 100 and traj_distances[wpNum - 1] < L:
        segment_end = wpNum - 1
    #     # print(traj_distances[-1])
    else:
        while traj_distances[segment_end] < L:
            segment_end = (segment_end + 1) % wpNum
    #     count += 1
    #     if count > wpNum:
    #         segment_end = wpNum - 1
    #         break
    segment_begin = (segment_end - 1 + wpNum) % wpNum
    x_array = np.linspace(waypoints[segment_begin, 0], waypoints[segment_end, 0], interpScale)
    y_array = np.linspace(waypoints[segment_begin, 1], waypoints[segment_end, 1], interpScale)
    v_array = np.linspace(waypoints[segment_begin, 2], waypoints[segment_end, 2], interpScale)
    xy_interp = np.vstack((x_array, y_array)).T
    dist_interp = simple_norm_axis1(xy_interp - curr_pos) - L
    i_interp = np.argmin(np.abs(dist_interp))
    target_global = np.array((x_array[i_interp], y_array[i_interp]))
    new_L = np.linalg.norm(curr_pos - target_global)
    return np.array((x_array[i_interp], y_array[i_interp], v_array[i_interp])), new_L, nearest_dist
