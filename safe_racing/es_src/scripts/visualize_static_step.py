import fire
import os
from es.planner.lattice_planner import LatticePlanner
from es.utils.visualize import draw_lattice_grid
from es.utils.utils import *
from es.utils.visualize import draw_pts, value2color, get_track_segment
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
def safe_plus(mod, start, delta):
    return (start + delta + mod) % mod


def random_position(waypoints, waypoints_xytheta, sampled_number=1, car_gap_idx=6, ego_idx=None):
    # TODO: the 25 term just for test, make sure waypoints are not at the beginning or end of the track
    if not ego_idx:
        ego_idx = random.sample(range(20, len(waypoints_xytheta) - 25), 1)[0]
        print(ego_idx)
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
    os.path.join(os.path.abspath('..'), 'configs', 'General1', 'lattice_planner_config.yaml')
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
# fake_poses, pos_idx = random_position(waypoints, waypoints_xytheta, num_agents)


####### GLOBAL VARIABLES #######
################ TOOL FUNCTION and VARIABLE #####################


def draw_traj_on_track(fake_v=6.4, with_cost=False):
    global wp_x, wp_y, waypoints, fake_poses, planner
    thres = 50
    wp_x = wp_x[pos_idx - thres:pos_idx + thres]
    wp_y = wp_y[pos_idx - thres:pos_idx + thres]
    waypoints = waypoints[pos_idx - thres:pos_idx + thres, :]
    ego_pose = fake_poses[0]
    all_traj, all_traj_clothoid, goal_grid = planner.generate_traj(ego_pose, fake_v, waypoints)
    xy_grid = goal_grid[:, :2]
    track_x, track_y = get_track_segment(planner, ego_pose, dist_thres=3.0)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(track_x, track_y, 'o', c='k', markersize=1.0, label='track')
    ax.plot(wp_x, wp_y, c='b', markersize=1.0, label='optimal raceline')
    ax.plot(ego_pose[0], ego_pose[1], 'ro', markersize=2.0, label='car_position')

    ax.plot(fake_poses[0][0], fake_poses[0][1], 'b', markersize=2.0, label='shortest distance')

    ### fake opponents
    for i in range(0, len(fake_poses)):
        if i != 0:
            ax.plot(fake_poses[i][0], fake_poses[i][1], 'go', markersize=3.0)
            oppo_vertices = get_vertices(fake_poses[i], length, width).T
            # print(oppo_vertices.shape)
            oppo_vertices = np.hstack((oppo_vertices, oppo_vertices[:, 0].reshape(2, -1)))
            draw_pts(oppo_vertices, ax, c='g', mksize=3.0, label='opponent car', pointonly=False, linewidth=3.0)
        else:
            ax.plot(fake_poses[i][0], fake_poses[i][1], 'ro', markersize=3.0)
            oppo_vertices = get_vertices(fake_poses[i], length, width).T
            oppo_vertices = np.hstack((oppo_vertices, oppo_vertices[:, 0].reshape(2, -1)))
            draw_pts(oppo_vertices, ax, c='r', mksize=3.0, label='ego car', pointonly=False, linewidth=3.0)
    ax.plot(np.array([fake_poses[0][0], fake_poses[1][0]]), np.array([fake_poses[0][1], fake_poses[1][1]]), 'k',
            linewidth=3.0, label='direction')
    draw_pts(xy_grid.T, ax, c='r', mksize=2.0, label='lattice points', pointonly=True)

    ### trajs
    n = len(all_traj)
    if not with_cost:
        draw_cost_colors, draw_cmapper = value2color(np.random.random(n))
        for i in range(n):
            draw_pts(np.array(all_traj[i]).T[:2], ax, c=draw_cost_colors[i], mksize=2.0, linewidth=1.0, pointonly=True)
            # if i % 2 == 0:
            #     ax.text(xy_grid[i][0], xy_grid[i][1], i)
    ax.axis('equal')
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.legend()
    plt.show()


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
    track_x, track_y = get_track_segment(planner, ego_pose, dist_thres=4.0)
    print("track_x = {}".format(track_x))
    print("track_y = {}".format(track_y))


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
    print(best_costs)
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

    cost_weights = np.array([
        2,
        2,
        2,
        5,
        2,
        2,
        0.15])
    planner.set_parameters({'cost_weights': cost_weights, 'traj_v_scale': 0.9})
    all_costs = planner.eval(all_traj, all_traj_clothoid, fake_poses[1:, :].reshape(-1, 3), ego_pose=ego_pose)  # (n, k)
    best_costs = np.min(all_costs, axis=1)
    print(best_costs)
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


def draw_traj_with_cost(fake_v=6.4):
    global wp_x, wp_y, waypoints, fake_poses, planner
    wp_x = wp_x[pos_idx - 25:pos_idx + 25]
    wp_y = wp_y[pos_idx - 25:pos_idx + 25]
    waypoints = waypoints[pos_idx - 50:pos_idx + 50, :]
    ego_pose = fake_poses[0]
    all_traj, all_traj_clothoid, goal_grid = planner.generate_traj(ego_pose, fake_v, waypoints)
    xy_grid = goal_grid[:, :2]
    track_x, track_y = get_track_segment(planner, ego_pose, dist_thres=8.0)

    all_costs = planner.eval(all_traj, all_traj_clothoid, fake_poses[1:, :].reshape(-1, 3), ego_pose=ego_pose)  # (n, k)
    best_costs = np.min(all_costs, axis=1)
    best_v_idx = np.argmin(all_costs, axis=1)
    best_v = all_traj[:, -1, 2] * planner.v_lattice_span[best_v_idx]
    print(f'v for each traj: {best_v}')
    best_traj_idx = np.argmin(all_costs)
    row_idx, col_idx = divmod(best_traj_idx, planner.v_lattice_num)
    best_traj = all_traj[row_idx]
    best_traj[:, 2] *= planner.v_lattice_span[col_idx]
    draw_cost_colors, draw_cmapper = value2color(best_costs)

    fig, (ax, ax1) = plt.subplots(1, 2, figsize=(20, 10))
    ax.plot(track_x, track_y, 'o', c='k', markersize=0.3, label='track')
    ax.plot(wp_x, wp_y, c='b', markersize=1.0, label='optimal raceline')
    ax.plot(ego_pose[0], ego_pose[1], 'ro', markersize=5.0, label='car_position')
    ### fake opponents
    for i in range(1, len(fake_poses)):
        ax.plot(fake_poses[i][0], fake_poses[i][1], 'go', markersize=5.0, label='opponent_position')
        oppo_vertices = get_vertices(fake_poses[i], length, width).T
        draw_pts(oppo_vertices, ax, c='g', mksize=5.0, label='lattice points', pointonly=True)

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
    for key, value in planner.step_all_cost.items():
        ax1.plot(cost_x, value, '-o', markersize=3.0, label=key)
        if len(planner.step_all_cost) > 1:
            for x, y in zip(cost_x, value):
                ax1.text(x, y, str(round(y, 1)), c='k', fontsize=10)
    ax1.plot(best_costs, '-o', markersize=3.0, label='total_cost')
    fig.colorbar(draw_cmapper, orientation='horizontal', label='Cost')
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.legend()
    plt.show()


# if __name__ == '__main__':
#     fire.Fire({
#         'global_traj': draw_traj_on_track,
#         'traj_v': draw_traj_with_v,
#         'traj_cost': draw_traj_with_cost
#     })

draw_traj_with_v()
