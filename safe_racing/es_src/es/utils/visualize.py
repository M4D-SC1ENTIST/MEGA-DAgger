import matplotlib.pyplot as plt

from es.utils.utils import *
from pyclothoids import Clothoid
import matplotlib
import matplotlib.cm as cm
import numpy as np

# try:
import pandas as pd
import seaborn as sns
from pyglet.gl import GL_POINTS
# except:
#     pass
from es.planner.lattice_planner import traj_global2local


class LaneSwitcherRender:
    def __init__(self, planner):
        self.planner = planner
        self.lane_pos = self.planner.lane_pos
        self.num_lanes = self.planner.num_lanes
        self.draw_lanes = [[] for _ in range(self.num_lanes)]

    def render_callback(self, e):
        x = e.cars[0].vertices[::2]
        y = e.cars[0].vertices[1::2]
        top, bottom, left, right = max(y), min(y), min(x), max(x)
        e.score_label.x = left
        e.score_label.y = top - 700
        e.left = left - 400
        e.right = right + 400
        e.top = top + 400
        e.bottom = bottom - 400

        for i in range(self.num_lanes):
            points = np.vstack((self.lane_pos[i][:, 0], self.lane_pos[i][:, 1])).T
            scaled_points = 50. * points
            for j in range(points.shape[0]):
                if len(self.draw_lanes[i]) < points.shape[0]:
                    b = e.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_points[j, 0], scaled_points[j, 1], 0.]),
                                    ('c3B/stream', [183, 193, 222]))
                    self.draw_lanes[i].append(b)
                else:
                    # self.draw_lanes[i].vertices = [scaled_points[j, 0], scaled_points[j, 1], 0.]
                    pass


class LatticePlannerRender:
    def __init__(self, planner):
        self.planner = planner
        self.draw_grid_pts = []
        self.draw_traj_pts = []
        self.draw_target = []
        self.draw_waypoints = []
        self.waypoints = self.planner.waypoints
        self.waypoints_xytheta = np.hstack((self.waypoints[:, :2], self.waypoints[:, 3].reshape(-1, 1)))

    def render_callback(self, e):
        """
        Custom render call back function for Lattice Planner General1

        Args:
            e: environment renderer
        """

        # update camera to follow car
        x = e.cars[0].vertices[::2]
        y = e.cars[0].vertices[1::2]
        top, bottom, left, right = max(y), min(y), min(x), max(x)
        e.score_label.x = left
        e.score_label.y = top - 700
        e.left = left - 400
        e.right = right + 400
        e.top = top + 400
        e.bottom = bottom - 400

        scaled_points = 50. * self.waypoints_xytheta[:, :2]


        if self.planner.goal_grid is not None:
            goal_grid_pts = np.vstack([self.planner.goal_grid[:, 0], self.planner.goal_grid[:, 1]]).T
            scaled_grid_pts = 50. * goal_grid_pts
            for i in range(scaled_grid_pts.shape[0]):
                if len(self.draw_grid_pts) < scaled_grid_pts.shape[0]:
                    b = e.batch.add(1, GL_POINTS, None,
                                    ('v3f/stream', [scaled_grid_pts[i, 0], scaled_grid_pts[i, 1], 0.]),
                                    ('c3B/stream', [183, 193, 222]))
                    self.draw_grid_pts.append(b)
                else:
                    self.draw_grid_pts[i].vertices = [scaled_grid_pts[i, 0], scaled_grid_pts[i, 1], 0.]

            best_traj_pts = np.vstack([self.planner.best_traj[:, 0], self.planner.best_traj[:, 1]]).T
            scaled_btraj_pts = 50. * best_traj_pts
            for i in range(scaled_btraj_pts.shape[0]):
                if len(self.draw_traj_pts) < scaled_btraj_pts.shape[0]:
                    b = e.batch.add(1, GL_POINTS, None,
                                    ('v3f/stream', [scaled_btraj_pts[i, 0], scaled_btraj_pts[i, 1], 0.]),
                                    ('c3B/stream', [183, 193, 222]))
                    self.draw_traj_pts.append(b)
                else:
                    self.draw_traj_pts[i].vertices = [scaled_btraj_pts[i, 0], scaled_btraj_pts[i, 1], 0.]
        self.planner.tracker.render_waypoints(e)


def draw_pts(pts, ax, c='b', mksize=5.0, label=None, pointonly=False, linewidth=0.5):
    """
    pts: (2, n)
    """
    x = pts[0]
    y = pts[1]
    if pointonly:
        ax.plot(x, y, 'o', c=c, markersize=mksize, label=label)
    else:
        ax.plot(x, y, c=c, markersize=mksize, label=label, linewidth=linewidth)


def value2color(value):
    """
    value: np.ndarray (n, )
    """

    minima = np.min(value)
    maxima = np.max(value)
    rgba = []

    norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.cool)

    for v in value:
        rgba.append(mapper.to_rgba(v))

    return rgba, mapper


def get_track_segment(planner, ego_pose, dist_thres):
    map_img = planner.map_img
    map_r = planner.map_resolution
    map_ori_x = planner.map_metainfo[0]
    map_ori_y = planner.map_metainfo[1]
    track_coor = np.nonzero(map_img == 0)
    # axis transformation
    track_y = track_coor[0] * map_r + map_ori_y
    track_x = track_coor[1] * map_r + map_ori_x
    track_xy = np.vstack((track_x, track_y))  # (2, n)
    # filter track near the sampled fake car position
    track_idx = np.nonzero(np.linalg.norm(track_xy.T - ego_pose[:2], axis=1) < dist_thres)[0]
    track_x = track_x[track_idx]
    track_y = track_y[track_idx]
    return track_x, track_y


def label_point(x, y, val, ax, fontsize=10, is_int=False):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        if not is_int:
            ax.text(point['x'] + .02, point['y'], str(round(point['val'], 2)), fontsize=fontsize)
        else:
            ax.text(point['x'] + .02, point['y'], str(int(point['val'])), fontsize=fontsize)


def draw_lattice_grid(fake_poses, pos_idx, planner, waypoints, width=0.31, length=0.58, draw_global_traj=True):
    pass
    ############## Draw Local Trajectory ###################
    # else:
    #     fig, ax = plt.subplots(figsize=(20, 10))
    #     ego_pose1 = ego_pose
    #     local_traj1 = traj_global2local(ego_pose1, all_traj[..., :2])
    #     fake_prev_traj = local_traj1[5]
    #
    #     ego_pose2 = fake_poses[-1]
    #     all_traj2, _, _ = generate_traj(ego_pose2, planner, waypoints)
    #     local_traj2 = traj_global2local(ego_pose2, all_traj2[..., :2])
    #     similarity_cost = get_similarity_cost(all_traj2, None, None, ego_pose2, fake_prev_traj)
    #     draw_cost_colors, draw_cmapper = value2color(cost)
    #     print(f'the most similar trajectory index {np.argmin(similarity_cost)}')
    #     print(f'similarity cost {similarity_cost}')
    #
    #     n = len(all_traj)
    #     for i in range(n):
    #         draw_pts(np.array(local_traj1[i]).T, ax, c=draw_cost_colors[i], linewidth=1.0)
    #     for i in range(n):
    #         draw_pts(np.array(local_traj2[i]).T, ax, c='k', linewidth=1.0)
    #
    #     ax.axis('equal')
    #     fig.colorbar(draw_cmapper, orientation='horizontal', label='Cost')
    #     plt.legend()
    #     plt.show()
    ############## Draw Local Trajectory ###################


def get_poses_for_step(states, step):
    ego_pose = np.array([states['ego_x'][step], states['ego_y'][step], states['ego_theta'][step]])
    opp_pose = np.array([states['opp_x'][step], states['opp_y'][step], states['opp_theta'][step]])
    return np.vstack((ego_pose, opp_pose))


def draw_traj_with_state(waypoints_xytheta, planner, poses, ego_traj, opp_traj, state, state_name):
    ego_pose = poses[0]
    _, _, _, idx = nearest_point(ego_pose[:2], waypoints_xytheta[:, :2])
    wp_x = waypoints_xytheta[idx - 100:idx + 100, 0]
    wp_y = waypoints_xytheta[idx - 100:idx + 100, 1]

    track_x, track_y = get_track_segment(planner, ego_pose, dist_thres=20)

    fig, (ax, ax1) = plt.subplots(1, 2, figsize=(20, 10))
    ax.plot(track_x, track_y, 'o', c='k', markersize=0.3, label='track')
    ax.plot(wp_x, wp_y, c='b', markersize=1.0, label='optimal raceline')

    draw_pts(pts=ego_traj, ax=ax, mksize=1.0, label='ego_traj', c='r')
    draw_pts(pts=opp_traj, ax=ax, mksize=1.0, label='opp_traj', c='g')

    ax1.plot(state, '-ko', markersize=0.5, label=state_name)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.legend()

    plt.show()


def draw_traj_with_cost(waypoints_xytheta, planner, poses, ego_v, opp_v, length=0.58, width=0.31, clip_mapcost=True,
                        traj_id=-1):
    ego_pose = poses[0]
    opp_pose = poses[1]
    _, _, _, idx = nearest_point(ego_pose[:2], waypoints_xytheta[:, :2])
    wp_x = waypoints_xytheta[idx - 25:idx + 25, 0]
    wp_y = waypoints_xytheta[idx - 25:idx + 25, 1]

    all_traj, all_traj_clothoid, goal_grid = planner.generate_traj(ego_pose, ego_v)
    xy_grid = goal_grid[:, :2]
    track_x, track_y = get_track_segment(planner, ego_pose, dist_thres=8.0)
    all_costs_w_v = planner.eval(all_traj, all_traj_clothoid, opp_pose.reshape(1, 3), ego_pose=ego_pose)  # (n, k)

    all_costs_wo_v = np.min(all_costs_w_v, axis=1)
    if clip_mapcost:
        all_costs_wo_v = np.clip(all_costs_wo_v, a_min=0.0, a_max=40.0)
    best_v_idx = np.argmin(all_costs_w_v, axis=1)
    best_v = all_traj[:, -1, 2] * planner.v_lattice_span[best_v_idx] * planner.traj_v_scale

    best_traj_idx = np.argmin(all_costs_w_v)
    row_idx, col_idx = divmod(best_traj_idx, planner.v_lattice_num)
    best_traj = all_traj[row_idx]
    best_traj[:, 2] *= planner.v_lattice_span[col_idx]
    draw_cost_colors, draw_cmapper = value2color(all_costs_wo_v)

    fig, (ax, ax1) = plt.subplots(1, 2, figsize=(20, 10))
    ax.plot(track_x, track_y, 'o', c='k', markersize=0.3, label='track')
    ax.plot(wp_x, wp_y, c='b', markersize=1.0, label='optimal raceline')
    ax.plot(ego_pose[0], ego_pose[1], 'ro', markersize=5.0, label='car_position')
    ### fake opponents
    for i in range(len(poses)):
        car_vertices = get_vertices(poses[i], length, width).T
        if i != 0:
            ax.plot(poses[i][0], poses[i][1], 'go', markersize=5.0, label='opponent_position')
            draw_pts(car_vertices, ax, c='g', mksize=5.0, label='opponents', pointonly=True)
        else:
            ax.plot(poses[i][0], poses[i][1], 'ro', markersize=5.0, label='opponent_position')
            draw_pts(car_vertices, ax, c='r', mksize=5.0, label='ego', pointonly=True)

    draw_pts(xy_grid.T, ax, c='r', mksize=2.0, label='lattice points', pointonly=True)
    ### trajs
    n = len(all_traj)
    for i in range(n):
        draw_pts(np.array(all_traj[i]).T[:2], ax, c=draw_cost_colors[i], linewidth=1.0)
        if i % 2 == 0:
            ax.text(xy_grid[i][0], xy_grid[i][1], i)
        if i == best_traj_idx:
            ax.text(xy_grid[i][0], xy_grid[i][1], i, c='r')
    draw_pts(np.array(best_traj).T[:2], ax, c='k', linewidth=1.0)

    cost_x = np.arange(0, n, 1)
    if traj_id == -1:
        # ax1.plot(all_costs_wo_v, '-o', markersize=3.0, label='total_cost')
        ax1.plot(best_v, 'o', markersize=3.0, label='velocity')
        for key, value in planner.step_all_cost.items():
            if key not in ('abs_v_cost', 'collision_cost', 'get_map_collision'):
                ax1.plot(cost_x, value, '-o', markersize=3.0, label=key)
            elif key != 'get_map_collision':
                ax1.plot(cost_x, np.min(value, axis=1), '-o', markersize=3.0, label=key)
    else:
        ax1.plot(all_costs_w_v[traj_id], '-o', markersize=3.0, label='traj_cost_with_v')
        ax1.plot(planner.step_all_cost['abs_v_cost'][traj_id], '-o', markersize=3.0, label='abs_v_cost')
        ax1.plot(planner.step_all_cost['collision_cost'][traj_id], '-o', markersize=3.0, label='collision_cost')

    fig.colorbar(draw_cmapper, orientation='horizontal', label='Cost')
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.legend()
    plt.show()
