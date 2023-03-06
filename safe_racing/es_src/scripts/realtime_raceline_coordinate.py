import os
from PIL import Image
import yaml
# from scipy.ndimage import distance_transform_edt as edt
from es.utils.utils import *
import numpy as np


import fire
import os
from es.planner.lattice_planner import LatticePlanner
from es.utils.visualize import draw_lattice_grid
from es.utils.utils import *
# from safe_racing.es_src.es.utils.visualize import draw_pts, value2color, get_track_segment
from es.utils.visualize import draw_pts, value2color
import random
from argparse import Namespace
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

module = os.path.dirname(os.path.abspath(__file__))

sns.set_style('white')
sns.set_style('ticks')
sns.set_context('poster')


################ TOOL FUNCTION and VARIABLE #####################
def get_realtime_raceline_coordinate(conf, ego_pose, dist_threshold=2):
    # load map image
    map_img_path = os.path.splitext(conf.map_path)[0] + conf.map_ext
    map_img = np.array(Image.open(map_img_path).transpose(Image.FLIP_TOP_BOTTOM)).astype(np.float64)
    map_img[map_img <= 128.] = 0.
    map_img[map_img > 128.] = 255.
    # map_height = map_img.shape[0]
    # map_width = map_img.shape[1]

    # load map yaml
    with open(conf.map_path + '.yaml', 'r') as yaml_stream:
        map_metadata = yaml.safe_load(yaml_stream)
        map_resolution = map_metadata['resolution']
        origin = map_metadata['origin']

    orig_x = origin[0]
    orig_y = origin[1]
    # orig_s = np.sin(origin[2])
    # orig_c = np.cos(origin[2])

    # dt = map_resolution * edt(map_img)
    # map_metainfo = (orig_x, orig_y, orig_c, orig_s, map_height, map_width, map_resolution)

    track_coor = np.nonzero(map_img == 0)

    # axis transformation
    track_y = track_coor[0] * map_resolution + orig_y
    track_x = track_coor[1] * map_resolution + orig_x
    track_xy = np.vstack((track_x, track_y))  # (2, n)
    # print("track_xy.shape[1] = {}".format(track_xy.shape[1]))

    # # filter track near the sampled fake car position
    # track_idx = np.nonzero(np.linalg.norm(track_xy.T - ego_pose[:2], axis=1) < dist_threshold)[0]
    # track_x = track_x[track_idx]
    # track_y = track_y[track_idx]

    # calculate every track points' dot product with ego pos for further checking left or right
    dir_vec_rot_90 = np.array([[math.cos(ego_pose[2] - math.pi / 2.0)], [math.sin(ego_pose[2] - math.pi / 2.0)]])
    vec_track_ego = track_xy.T - np.tile(ego_pose[:2], (track_xy.shape[1], 1))
    dot_val = vec_track_ego @ dir_vec_rot_90
    # print(dot_val.shape)

    # calculate every track points' norm with ego pos
    vec_norm = np.linalg.norm(track_xy.T - ego_pose[:2], axis=1)
    vec_norm = vec_norm.reshape((len(vec_norm), 1))
    # print(vec_norm.shape)

    # form = np.hstack((dot_val, vec_norm, track_xy.T))
    # print(form.shape)

    left_min_norm = None
    right_min_norm = None
    left_track_pos = None
    right_track_pos = None

    for i in range(len(vec_norm)):
        if dot_val[i] > 0:  # left
            if left_min_norm is None or vec_norm[i] < left_min_norm:
                left_min_norm = vec_norm[i]
                left_track_pos = track_xy.T[i, :]
        else:
            if right_min_norm is None or vec_norm[i] < right_min_norm:
                right_min_norm = vec_norm[i]
                right_track_pos = track_xy.T[i, :]

    print(left_track_pos)
    print(right_track_pos)

    # track_x = np.array([left_track_pos[0], right_track_pos[0]])
    # track_y = np.array([left_track_pos[1], right_track_pos[1]])

    # return track_x, track_y
    return left_track_pos, right_track_pos  # only for Shuo's usage, cannot run in curr test example!


def safe_plus(mod, start, delta):
    return (start + delta + mod) % mod


def random_position(waypoints, waypoints_xytheta, sampled_number=1, car_gap_idx=6, ego_idx=None):
    # TODO: the 25 term just for test, make sure waypoints are not at the beginning or end of the track
    if not ego_idx:
        ego_idx = random.sample(range(20, len(waypoints_xytheta) - 25), 1)[0]
        # print(ego_idx)
    for i in range(sampled_number):
        starting_idx = (ego_idx + i * car_gap_idx) % len(waypoints_xytheta)
        x, y, theta = waypoints_xytheta[starting_idx][0], waypoints_xytheta[starting_idx][1], \
                      waypoints_xytheta[starting_idx][2]
        theta = (zero_2_2pi(theta) + 0.5 * np.pi) + random.random() * 0.1 * np.pi
        if i == 0:
            res = np.array([[x, y, theta]])  # (1, 3)
        else:
            res = np.vstack((res, np.array([[x, y, theta]])))
    return res, ego_idx


####### GLOBAL VARIABLES #######
width = 0.31  # 0.31
length = 0.58  # 0.58

# loading waypoints
lattice_planner_conf_path = \
    os.path.join('..', 'configs', 'General1', 'lattice_planner_config.yaml')
with open(lattice_planner_conf_path) as file:
    lp_conf_dict = yaml.load(file, Loader=yaml.FullLoader)
lp_conf = Namespace(**lp_conf_dict)
planner = LatticePlanner(lp_conf)
waypoints = planner.waypoints
waypoints_xytheta = np.hstack((waypoints[:, :2], waypoints[:, 3].reshape(-1, 1)))
wp_x = waypoints[:, 0]
wp_y = waypoints[:, 1]

## get fake pose
num_agents = 2


def draw_traj_with_v(fake_v=6.4):
    global wp_x, wp_y, waypoints, planner
    all_waypoints = waypoints
    fake_poses, pos_idx = random_position(waypoints, waypoints_xytheta, sampled_number=2, ego_idx=360, car_gap_idx=8)
    thres = 20
    wp_x = wp_x[pos_idx - thres:pos_idx + thres]
    wp_y = wp_y[pos_idx - thres:pos_idx + thres]
    waypoints = waypoints[pos_idx - thres:pos_idx + thres, :]
    ego_pose = fake_poses[0]
    all_traj, all_traj_clothoid, goal_grid = planner.generate_traj(ego_pose, fake_v, all_waypoints)
    xy_grid = goal_grid[:, :2]
    track_x, track_y = get_realtime_raceline_coordinate(lp_conf, ego_pose, dist_threshold=2.0)
    # print("track_x = {}".format(track_x))
    # print("track_y = {}".format(track_y))

    cost_weights = np.array([
        2,
        2,
        2,
        2,
        2,
        2,
        1])
    planner.set_parameters({'cost_weights': cost_weights, 'traj_v_scale': 0.9})
    all_costs = planner.eval(all_traj, all_traj_clothoid, fake_poses[1:, :].reshape(-1, 3), ego_pose=ego_pose)  # (n, k)
    best_costs = np.min(all_costs, axis=1)
    # print(best_costs)
    best_v_idx = np.argmin(all_costs, axis=1)
    best_v = all_traj[:, -1, 2] * planner.v_lattice_span[best_v_idx]
    best_traj_idx = np.argmin(all_costs)
    row_idx, col_idx = divmod(best_traj_idx, planner.v_lattice_num)
    best_traj = all_traj[row_idx]
    best_traj[:, 2] *= planner.v_lattice_span[col_idx]
    draw_cost_colors, draw_cmapper = value2color(best_costs)

    # fig, (ax, ax1) = plt.subplots(1, 2, figsize=(20, 10))
    # fig, ax = plt.subplots(figsize=(10, 10))
    fig = plt.figure()
    gs = fig.add_gridspec(1, 2, hspace=0, wspace=0)
    ax = gs.subplots(sharex=True, sharey=False)
    ax, ax1 = ax

    ax.plot(track_x, track_y, 'o', c='k', markersize=1.0, label='track')
    ax.plot(wp_x, wp_y, c='b', linewidth=1.0, label='optimal raceline')
    # ax.plot(ego_pose[0], ego_pose[1], 'ro', markersize=5.0, label='car_position')
    ### fake opponents
    for i in range(0, len(fake_poses)):
        if i != 0:
            ax.plot(fake_poses[i][0], fake_poses[i][1], 'ro', markersize=5.0)
            oppo_vertices = get_vertices(fake_poses[i], length, width).T
            # print(oppo_vertices.shape)
            oppo_vertices = np.hstack((oppo_vertices, oppo_vertices[:, 0].reshape(2, -1)))
            draw_pts(oppo_vertices, ax, c='r', mksize=5.0, label='opponent car', pointonly=False, linewidth=5.0)
        else:
            ax.plot(fake_poses[i][0], fake_poses[i][1], 'go', markersize=5.0)
            oppo_vertices = get_vertices(fake_poses[i], length, width).T
            oppo_vertices = np.hstack((oppo_vertices, oppo_vertices[:, 0].reshape(2, -1)))
            draw_pts(oppo_vertices, ax, c='b', mksize=5.0, label='ego car', pointonly=False, linewidth=5.0)

    draw_pts(xy_grid.T, ax, c='r', mksize=5.0, label='lattice points', pointonly=True)
    ### trajs
    n = len(all_traj)
    for i in range(n):
        draw_pts(np.array(all_traj[i]).T[:2], ax, c=draw_cost_colors[i], linewidth=1.5)
        # if i % 2 == 0:
        #     ax.text(xy_grid[i][0], xy_grid[i][1], i)
        # if i == best_traj_idx:
        #     ax.text(xy_grid[i][0], xy_grid[i][1], i, c='r', fontsize=5)
    draw_pts(np.array(best_traj).T[:2], ax, c='k', linewidth=4.0, label='best trajectory')

    # ax1.plot(best_v, '-o', markersize=3.0, label='velocity')
    ax.axis('equal')
    # fig.colorbar(draw_cmapper, label='Cost')
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.9, 0.11, 0.02, 0.77])
    fig.colorbar(draw_cmapper, label='Cost', cax=cbar_ax)
    ax.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False)  # labels along the bottom edge are off
    ax.legend()

    cost_weights = np.array([2, 2, 2, 5, 2, 2, 0.15])
    planner.set_parameters({'cost_weights': cost_weights, 'traj_v_scale': 0.9})
    all_costs = planner.eval(all_traj, all_traj_clothoid, fake_poses[1:, :].reshape(-1, 3), ego_pose=ego_pose)  # (n, k)
    best_costs = np.min(all_costs, axis=1)
    # print(best_costs)
    best_v_idx = np.argmin(all_costs, axis=1)
    best_v = all_traj[:, -1, 2] * planner.v_lattice_span[best_v_idx]
    best_traj_idx = np.argmin(all_costs)
    row_idx, col_idx = divmod(best_traj_idx, planner.v_lattice_num)
    best_traj = all_traj[row_idx]
    best_traj[:, 2] *= planner.v_lattice_span[col_idx]
    draw_cost_colors, draw_cmapper = value2color(best_costs)

    ax1.plot(track_x, track_y, 'o', c='k', markersize=1.0, label='track')
    ax1.plot(wp_x, wp_y, c='b', linewidth=1.0, label='optimal raceline')
    # ax.plot(ego_pose[0], ego_pose[1], 'ro', markersize=5.0, label='car_position')
    ### fake opponents
    for i in range(0, len(fake_poses)):
        if i != 0:
            ax1.plot(fake_poses[i][0], fake_poses[i][1], 'ro', markersize=5.0)
            oppo_vertices = get_vertices(fake_poses[i], length, width).T
            # print(oppo_vertices.shape)
            oppo_vertices = np.hstack((oppo_vertices, oppo_vertices[:, 0].reshape(2, -1)))
            draw_pts(oppo_vertices, ax1, c='r', mksize=5.0, label='opponent car', pointonly=False, linewidth=5.0)
        else:
            ax1.plot(fake_poses[i][0], fake_poses[i][1], 'go', markersize=5.0)
            oppo_vertices = get_vertices(fake_poses[i], length, width).T
            oppo_vertices = np.hstack((oppo_vertices, oppo_vertices[:, 0].reshape(2, -1)))
            draw_pts(oppo_vertices, ax1, c='b', mksize=5.0, label='ego car', pointonly=False, linewidth=5.0)

    draw_pts(xy_grid.T, ax1, c='r', mksize=4.0, label='lattice points', pointonly=True)
    ### trajs
    n = len(all_traj)
    for i in range(n):
        draw_pts(np.array(all_traj[i]).T[:2], ax1, c=draw_cost_colors[i], linewidth=1.5)
        # if i % 2 == 0:
        #     ax.text(xy_grid[i][0], xy_grid[i][1], i)
        # if i == best_traj_idx:
        #     ax.text(xy_grid[i][0], xy_grid[i][1], i, c='r', fontsize=5)
    draw_pts(np.array(best_traj).T[:2], ax1, c='k', linewidth=5.0, label='best trajectory')

    ax.text(-30.5, 0.9, "velocity of best trajectory: 8 m/s", size=20.0, rotation=0.,
            ha="center", va="center",
            # bbox=dict(boxstyle="round",
            #           # ec=(0., 0., 0.),
            #           fc=(1., 1., 1.),
            #           )
            )
    ax1.text(-30.5, 0.9, "velocity of best trajectory: 4 m/s", size=20.0, rotation=0.,
             ha="center", va="center")

    # ax1.plot(best_v, '-o', markersize=3.0, label='velocity')
    ax1.axis('equal')
    ax1.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False)  # labels along the bottom edge are off
    # plt.rc('xtick', labelsize=20)
    # plt.rc('ytick', labelsize=20)
    ax.set_title('Weights=[2, 2, 2, 2, 2, 2, 2]')
    ax1.set_title('Weights=[2, 2, 2, 4, 2, 2, 1]')

    plt.show()


# draw_traj_with_v()
