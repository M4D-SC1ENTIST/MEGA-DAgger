import os

import gym
import numpy as np

from es.planner.lattice_planner import *
from es.utils.utils import *
from es.utils.DataProcessor import obsDict2carStateSeq
from es.worker import calculate_objectives
from datetime import datetime
from es.utils.visualize import LatticePlannerRender
logger = logging.getLogger(__name__)
from es.utils.DataProcessor import RolloutDataLogger
"""
waypoints: [x, y, v, heading, kappa]
"""

from pyglet.gl import GL_POINTS


def main():
    """
    Lattice Planner example. This example uses fixed waypoints throughout the 2 laps.
    For an example using dynamic waypoints, see the lane switcher example.
    """
    global ego_planner
    global draw_grid_pts
    global draw_traj_pts
    global draw_target
    global draw_waypoints
    global waypoints_xytheta
    # logger
    now = datetime.now()
    now_str = now.strftime("%m_%d_%H_%M_%S")
    log_path = f'./logger/lattice_planner{now_str}'
    # init_log_writer('INFO', log_path)

    ## state setting
    seed = 6300
    rng = np.random.default_rng(seed)
    xy_noise_scale = 0.0
    theta_noise_scale = 0.0

    map_name = 'map0'
    # map_name = 'General1'
    lattice_planner_conf_path = \
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', map_name, 'lattice_planner_config.yaml')
    with open(lattice_planner_conf_path) as file:
        lp_conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    lp_conf = Namespace(**lp_conf_dict)
    ego_planner = LatticePlanner(lp_conf)
    cost_weights = np.array([
        1,
        1,
        1,
        1,
        1,
        1,
        1])
    ego_planner.set_parameters({'cost_weights': cost_weights, 'traj_v_scale': 0.9})
    opp_planner = LatticePlanner(lp_conf)
    opp_planner.set_parameters({'cost_weights': cost_weights, 'traj_v_scale': 0.8})
    waypoints = ego_planner.waypoints
    waypoints_xytheta = np.hstack((waypoints[:, :2], waypoints[:, 3].reshape(-1, 1)))
    wpt_xyhs = np.vstack(
            (ego_planner.waypoints[:, 0], ego_planner.waypoints[:, 1],
             ego_planner.waypoints[:, 3], ego_planner.waypoints[:, 4])).T
    s_max = waypoints[-1, 4]

    # rendering
    render = LatticePlannerRender(ego_planner)

    # create environment
    num_agents = 2
    env = gym.make('f110_gym:f110-v0', map=lp_conf.map_path, map_ext='.png', num_agents=num_agents)
    env.add_render_callback(render.render_callback)
    logger = RolloutDataLogger(rollout_states=lp_conf.rollout_states, log_location=None)
    random_agent_pos, ego_idx = random_position(waypoints_xytheta, num_agents, rng, xy_noise_scale, theta_noise_scale)

    obs, _, done, _ = env.reset(random_agent_pos)
    env.render('human_fast')

    opp_planner.init_rollout()
    ego_planner.init_rollout()
    laptime = 0
    ego_s = 0.0
    opp_s = 0.0

    rollout_step = 0

    ### init for a rollout ###
    while not done and laptime < 60.0:
        rollout_step += 1
        ######### lattice plan ########
        oppo_pose = obsDict2oppoArray(obs, 0)
        ego_best_traj = ego_planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], oppo_pose,
                                         obs['linear_vels_x'][0])

        oppo_pose = obsDict2oppoArray(obs, 1)
        opp_best_traj = opp_planner.plan(obs['poses_x'][1], obs['poses_y'][1], obs['poses_theta'][1], oppo_pose,
                                         obs['linear_vels_x'][1])

        tracker_count = 0
        while not done and tracker_count < lp_conf.tracker_steps:
            ego_steer, ego_speed = ego_planner.tracker.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0],
                                                            obs['linear_vels_x'][0], ego_best_traj)
            opp_steer, opp_speed = opp_planner.tracker.plan(obs['poses_x'][1], obs['poses_y'][1], obs['poses_theta'][1],
                                                            obs['linear_vels_x'][1], opp_best_traj)
            action = np.array([[ego_steer, ego_speed], [opp_steer, opp_speed]])
            obs, timestep, done, _ = env.step(action)
            if np.any(obs['collisions']):
                done = True
            if tracker_count != 0:
                step_states = {
                    'ego_control_error': ego_planner.tracker.nearest_dist
                }
                logger.update_buffer(step_states)
            laptime += timestep
            tracker_count += 1
            env.render('human')
        # progress
        ego_s = ego_planner.cal_s()
        opp_s = opp_planner.cal_s()

        if ego_s > opp_s and rollout_step < 2:
            opp_s = opp_s + ego_planner.s_max
            opp_planner.last_s = opp_s

    print('Sim elapsed time:', laptime)


def random_position(waypoints_xytheta, sampled_number=1, rng=None, xy_noise=0.0, theta_noise=0.0):
    # 632, 550, 1408 overtake
    # 1041, 832 near
    # 380
    # ego_idx = 832
    ego_idx = rng.choice(np.arange(0, len(waypoints_xytheta)), 1)[0]
    # ego_idx = 300
    # ego_idx = 1508 222, 440
    print(f'ego_idx is {ego_idx}')
    for i in range(sampled_number):
        starting_idx = (ego_idx - i * 10) % len(waypoints_xytheta)
        x, y, theta = waypoints_xytheta[starting_idx][0], waypoints_xytheta[starting_idx][1],waypoints_xytheta[starting_idx][2]
        x = x + rng.random(size=1)[0] * xy_noise
        y = y + rng.random(size=1)[0] * xy_noise
        theta = (zero_2_2pi(theta) + 0.5 * np.pi) + rng.random(size=1)[0] * theta_noise
        if i == 0:
            res = np.array([[x, y, theta]])  # (1, 3)
        else:
            res = np.vstack((res, np.array([[x, y, theta]])))
    return res, ego_idx


def render_callback(e):
    """
    Custom render call back function for Lattice Planner General1

    Args:
        e: environment renderer
    """

    global ego_planner
    global draw_grid_pts
    global draw_traj_pts
    global draw_target
    global draw_waypoints
    global waypoints_xytheta

    # update camera to follow car
    x = e.cars[0].vertices[::2]
    y = e.cars[0].vertices[1::2]
    top, bottom, left, right = max(y), min(y), min(x), max(x)
    e.score_label.x = left
    e.score_label.y = top - 700
    e.left = left - 600
    e.right = right + 600
    e.top = top + 600
    e.bottom = bottom - 600

    scaled_points = 50. * waypoints_xytheta[:, :2]

    if ego_planner.goal_grid is not None:
        goal_grid_pts = np.vstack([ego_planner.goal_grid[:, 0], ego_planner.goal_grid[:, 1]]).T
        scaled_grid_pts = 50. * goal_grid_pts
        for i in range(scaled_grid_pts.shape[0]):
            if len(draw_grid_pts) < scaled_grid_pts.shape[0]:
                b = e.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_grid_pts[i, 0], scaled_grid_pts[i, 1], 0.]),
                                ('c3B/stream', [183, 193, 222]))
                draw_grid_pts.append(b)
            else:
                draw_grid_pts[i].vertices = [scaled_grid_pts[i, 0], scaled_grid_pts[i, 1], 0.]

        best_traj_pts = np.vstack([ego_planner.best_traj[:, 0], ego_planner.best_traj[:, 1]]).T
        scaled_btraj_pts = 50. * best_traj_pts
        for i in range(scaled_btraj_pts.shape[0]):
            if len(draw_traj_pts) < scaled_btraj_pts.shape[0]:
                b = e.batch.add(1, GL_POINTS, None,
                                ('v3f/stream', [scaled_btraj_pts[i, 0], scaled_btraj_pts[i, 1], 0.]),
                                ('c3B/stream', [183, 193, 222]))
                draw_traj_pts.append(b)
            else:
                draw_traj_pts[i].vertices = [scaled_btraj_pts[i, 0], scaled_btraj_pts[i, 1], 0.]
    ego_planner.tracker.render_waypoints(e)


if __name__ == '__main__':
    main()
