import time
import yaml
import gym
import numpy as np
from argparse import Namespace
from es.planner.pure_pursuit import AdvancedPurePursuitPlanner
import random
from es.utils.utils import zero_2_2pi
import os

def random_position(waypoints_xytheta=None):

    # 632, 550, 1408 overtake
    # 1041, 832 near
    # ego_idx = 832
    ego_idx = random.sample(range(len(waypoints_xytheta)), 1)[0]
    ego_idx = 1
    print(f'ego_idx is {ego_idx}')
    x, y, theta = waypoints_xytheta[ego_idx][0], waypoints_xytheta[ego_idx][1], \
                  waypoints_xytheta[ego_idx][2]
    # -np.pi, np.pi
    theta = (zero_2_2pi(theta) - 0.5*np.pi + np.pi)
    res = np.array([[x, y, theta]])  # (1, 3)
    return res


def main():
    mapname = 'map0'
    conf_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', mapname, 'pure_pursuit_config.yaml')
    work = {'mass': 3.463388126201571, 'lf': 0.15597534362552312, 'tlad': 0.82461887897713965,
            'vgain': 0.90338203837889}

    with open(conf_file) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    planner = AdvancedPurePursuitPlanner(conf, 0.17145+0.15875)
    waypoints_xytheta = np.vstack((planner.waypoints[:, 0], planner.waypoints[:, 1], planner.waypoints[:, 3])).T
    print(len(waypoints_xytheta))

    def render_callback(env_renderer):
        # custom extra drawing function

        e = env_renderer

        # update camera to follow car
        x = e.cars[0].vertices[::2]
        y = e.cars[0].vertices[1::2]
        top, bottom, left, right = max(y), min(y), min(x), max(x)
        e.score_label.x = left
        e.score_label.y = top - 700
        e.left = left - 500
        e.right = right + 500
        e.top = top + 500
        e.bottom = bottom - 500

        planner.render_waypoints(env_renderer)

    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1)
    env.add_render_callback(render_callback)

    obs, step_reward, done, info = env.reset(random_position(waypoints_xytheta))
    env.render()

    laptime = 0.0
    start = time.time()

    tol_error = 0
    step = 0
    while not done and step < 10000:
        steer, speed = planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], obs['linear_vels_x'][0], planner.waypoints)
        obs, step_reward, done, info = env.step(np.array([[steer, speed]]))
        tol_error += np.abs(planner.nearest_dist)
        laptime += step_reward
        env.render(mode='human_fast')
        step += 1

    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time() - start)
    print(tol_error)


if __name__ == '__main__':
    main()