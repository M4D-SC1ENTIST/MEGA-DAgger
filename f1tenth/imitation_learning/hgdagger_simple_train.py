import torch
import gym
import numpy as np
import argparse
import yaml

from il_utils.policies.agents.agent_mlp import AgentPolicyMLP
from il_utils.policies.experts.expert_waypoint_follower import ExpertWaypointFollower

import il_utils.utils.env_utils as env_utils

# hg-dagger import
from il_utils.dataset import Dataset
from il_utils.utils import agent_utils
from pathlib import Path


def process_parsed_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--algorithm', type=str, default='hg-dagger', help='imitation learning algorithm to use')
    # arg_parser.add_argument('--training_config', type=str, required=True,
    #                         help='the yaml file containing the training configuration')
    arg_parser.add_argument('--training_config', type=str, default='il_utils/il_config.yaml',
                            help='the yaml file containing the training configuration')
    return arg_parser.parse_args()


def initialization(il_config):
    seed = il_config['random_seed']
    # np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the environment
    map_conf = None

    if not il_config['environment']['random_generation']:
        if il_config['environment']['map_config_location'] is None:
            # If no environment is specified but random generation is off, use the default gym environment
            with open('il_utils/map/example_map/config_example_map.yaml') as file:
                map_conf_dict = yaml.load(file, Loader=yaml.FullLoader)
        else:
            # If an environment is specified and random generation is off, use the specified environment
            with open(il_config['environment']['map_config_location']) as file:
                map_conf_dict = yaml.load(file, Loader=yaml.FullLoader)
        map_conf = argparse.Namespace(**map_conf_dict)
        env = gym.make('f110_gym:f110-v0', map=map_conf.map_path, map_ext=map_conf.map_ext, num_agents=1)
        env.add_render_callback(env_utils.render_callback)
    else:
        # TODO: If random generation is on, generate random environment
        pass

    # obs, step_reward, done, info = env.reset(np.array([[map_conf.sx, map_conf.sy, map_conf.stheta]]))

    # Initialize the agent
    if il_config['policy_type']['agent']['model'] == 'mlp':
        agent = AgentPolicyMLP(il_config['policy_type']['agent']['observation_shape'],
                               il_config['policy_type']['agent']['hidden_dim'],
                               2,
                               il_config['policy_type']['agent']['learning_rate'],
                               device)
    else:
        # TODO: Implement other model (Transformer)
        pass

    # Initialize the expert
    if il_config['policy_type']['expert']['behavior'] == 'waypoint_follower':
        expert = ExpertWaypointFollower(map_conf)
    else:
        # TODO: Implement other expert behavior (Lane switcher and hybrid)
        pass

    start_pose = np.array([[map_conf.sx, map_conf.sy, map_conf.stheta]])
    # observation_gap = int(1080/il_config['policy_type']['agent']['observation_shape'])
    observation_shape = il_config['policy_type']['agent']['observation_shape']
    downsampling_method = il_config['policy_type']['agent']['downsample_method']

    render = il_config['environment']['render']
    render_mode = il_config['environment']['render_mode']

    return seed, agent, expert, env, start_pose, observation_shape, downsampling_method, render, render_mode


# Behavioral Cloning for bootstrap
def bc(seed, agent, expert, env, start_pose, observation_shape, downsampling_method, render, render_mode, purpose):
    best_model_saving_threshold = 500000

    algo_name = "BehavioralCloning"
    best_model = agent
    longest_distance_travelled = 0

    # For Sim2Real
    path = "il_utils/models/{}".format(algo_name)
    num_of_saved_models = 0

    resume_pose = start_pose
    is_last_round_done = False

    if render:
        eval_batch_size = 1
    else:
        eval_batch_size = 10

    max_traj_len = 10000

    if purpose == "train":
        n_iter = 100
    elif purpose == "bootstrap":
        n_iter = 1
    else:
        raise ValueError("purpose must be either 'train' or 'bootstrap'")

    num_of_samples_increment = 500

    n_batch_updates_per_iter = 1000

    train_batch_size = 64

    # np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = Dataset()

    log = {'Number of Samples': [],
           'Number of Expert Queries': [],
           'Mean Distance Travelled': [],
           'STDEV Distance Travelled': [],
           'Mean Reward': [],
           'STDEV Reward': []}

    # Perform BC
    for iter in range(n_iter + 1):
        if purpose == "train":
            print("-" * 30 + ("\ninitial:" if iter == 0 else "\niter {}:".format(iter)))
        else:
            print("- " * 15 + "\nbootstrap using BC:")

        # Evaluate the agent's performance
        # No evaluation at the initial iteration
        if iter > 0:
            print("Evaluating agent...")
            print("- " * 15)
            mean_travelled_distances, stdev_travelled_distances, mean_reward, stdev_reward = \
                agent_utils.eval(env, agent, start_pose, max_traj_len, eval_batch_size, observation_shape,
                                 downsampling_method, render, render_mode)

            log['Mean Distance Travelled'].append(mean_travelled_distances)
            log['STDEV Distance Travelled'].append(stdev_travelled_distances)
            log['Mean Reward'].append(mean_reward)
            log['STDEV Reward'].append(stdev_reward)

            # Replace the best model if the current model is better
            if (log['Mean Distance Travelled'][-1] > longest_distance_travelled) and (
                    log['Number of Samples'][-1] < best_model_saving_threshold):
                longest_distance_travelled = log['Mean Distance Travelled'][-1]
                best_model = agent

            # For Sim2Real
            if log['Mean Distance Travelled'][-1] > 100:
                curr_dist = log['Mean Distance Travelled'][-1]
                current_expsamples = log['Number of Expert Queries'][-1]
                model_path = Path(
                    path + f'/{algo_name}_svidx_{str(num_of_saved_models)}_dist_{int(curr_dist)}_expsamp_{int(current_expsamples)}.pkl')
                model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(agent.state_dict(), model_path)
                num_of_saved_models += 1

            print("Number of Samples: {}".format(log['Number of Samples'][-1]))
            print("Number of Expert Queries: {}".format(log['Number of Expert Queries'][-1]))
            print("Distance Travelled: {} (+/- {})".format(log['Mean Distance Travelled'][-1],
                                                           log['STDEV Distance Travelled'][-1]))
            print("Reward: {} (+/- {})".format(log['Mean Reward'][-1], log['STDEV Reward'][-1]))

            print("- " * 15)

            # DELETE IT WHEN DOING SIM2REAL
            if log['Number of Samples'][-1] > 3000:
                break

        if iter == n_iter:
            break

        tlad = 0.82461887897713965
        vgain = 0.90338203837889

        # Collect data from the expert
        print("Collecting data from the expert...")
        traj = {"observs": [], "poses_x": [], "poses_y": [], "poses_theta": [], "scans": [], "actions": [], "reward": 0}
        done = False
        obs, step_reward, done, info = env.reset(resume_pose)

        # Start rendering
        if render:
            if env.renderer is None:
                env.render()

        if purpose == "train":
            step_num = num_of_samples_increment
        else:
            step_num = 500

        for j in range(step_num):
            traj["observs"].append(obs)
            scan = agent_utils.downsample_and_extract_lidar(obs, observation_shape, downsampling_method)

            # Add Sim2Real noise
            sim2real_noise = np.random.uniform(-0.25, 0.25, scan.shape)
            scan = scan + sim2real_noise

            traj["scans"].append(scan)
            traj["poses_x"].append(obs["poses_x"][0])
            traj["poses_y"].append(obs["poses_y"][0])
            traj["poses_theta"].append(obs["poses_theta"][0])

            speed, steer = expert.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], tlad, vgain)
            action = np.array([[steer, speed]])

            obs, step_reward, done, info = env.step(action)

            # Update rendering
            if render:
                env.render(mode=render_mode)

            traj["actions"].append(action)
            traj["reward"] += step_reward

            if done:
                is_last_round_done = True
                break

        # To evenly sampling using expert by resuming at the last pose in the next iteration
        if is_last_round_done:
            resume_pose = start_pose
        else:
            resume_pose = np.array([[obs["poses_x"][0], obs["poses_y"][0], obs["poses_theta"][0]]])

        traj["observs"] = np.vstack(traj["observs"])
        traj["poses_x"] = np.vstack(traj["poses_x"])
        traj["poses_y"] = np.vstack(traj["poses_y"])
        traj["poses_theta"] = np.vstack(traj["poses_theta"])
        traj["scans"] = np.vstack(traj["scans"])
        traj["actions"] = np.vstack(traj["actions"])

        # Adding to datasets
        print("Adding to dataset...")
        dataset.add(traj)

        log['Number of Samples'].append(dataset.get_num_of_total_samples())
        log['Number of Expert Queries'].append(dataset.get_num_of_total_samples())

        # Train the agent
        print("Training agent...")
        for _ in range(n_batch_updates_per_iter):
            train_batch = dataset.sample(train_batch_size)
            agent.train(train_batch["scans"], train_batch["actions"])

        if purpose == "bootstrap":
            return agent, log, dataset

    # Save log and the best model
    agent_utils.save_log_and_model(log, best_model, algo_name)


# HG-DAgger for training
def hg_dagger(seed, agent, expert, env, start_pose, observation_shape, downsampling_method, render, render_mode):
    algo_name = "HGDAgger"
    best_model = agent
    longest_distance_travelled = 0

    num_of_expert_queries = 0

    # For Sim2Real
    path = "models/{}".format(algo_name)
    num_of_saved_models = 0

    if render:
        eval_batch_size = 1
    else:
        eval_batch_size = 10

    init_traj_len = 50
    max_traj_len = 3500
    n_batch_updates_per_iter = 1000

    eval_max_traj_len = 10000

    train_batch_size = 64

    # np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = Dataset()

    log = {'Number of Samples': [],
           'Number of Expert Queries': [],
           'Mean Distance Travelled': [],
           'STDEV Distance Travelled': [],
           'Mean Reward': [],
           'STDEV Reward': []}

    # Perform HG-DAgger
    n_iter = 267  # Number of Epochs

    n_rollout = 5

    tlad = 0.82461887897713965
    vgain = 0.90338203837889

    # Epochs
    for iter in range(n_iter + 1):
        print("-" * 30 + ("\ninitial:" if iter == 0 else "\niter {}:".format(iter)))
        # Evaluation
        if iter > 0:
            print("Evaluating agent...")
            print("- " * 15)
            # log["Iteration"].append(iter)
            mean_travelled_distances, stdev_travelled_distances, mean_reward, stdev_reward = \
                agent_utils.eval(env, agent, start_pose, eval_max_traj_len, eval_batch_size, observation_shape,
                                 downsampling_method, render, render_mode)

            log['Mean Distance Travelled'].append(mean_travelled_distances)
            log['STDEV Distance Travelled'].append(stdev_travelled_distances)
            log['Mean Reward'].append(mean_reward)
            log['STDEV Reward'].append(stdev_reward)

            # Replace the best model if the current model is better
            if log['Mean Distance Travelled'][-1] > longest_distance_travelled:
                longest_distance_travelled = log['Mean Distance Travelled'][-1]
                best_model = agent

            # For Sim2Real
            if log['Mean Distance Travelled'][-1] > 100:
                curr_dist = log['Mean Distance Travelled'][-1]
                current_expsamples = log['Number of Expert Queries'][-1]
                model_path = Path(
                    path + f'/{algo_name}_svidx_{str(num_of_saved_models)}_dist_{int(curr_dist)}_expsamp_{int(current_expsamples)}.pkl')
                model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(agent.state_dict(), model_path)
                num_of_saved_models += 1

            print("Number of Samples: {}".format(log['Number of Samples'][-1]))
            print("Number of Expert Queries: {}".format(log['Number of Expert Queries'][-1]))
            print("Distance Travelled: {} (+/- {})".format(log['Mean Distance Travelled'][-1],
                                                           log['STDEV Distance Travelled'][-1]))
            print("Reward: {} (+/- {})".format(log['Mean Reward'][-1], log['STDEV Reward'][-1]))

            print("- " * 15)

            # DELETE IT WHEN DOING SIM2REAL
            if log['Number of Expert Queries'][-1] > 3000:
                break

        if iter == n_iter:
            break

        if iter == 0:
            # Bootstrap using BC
            agent, log, dataset = bc(seed, agent, expert, env, start_pose, observation_shape, downsampling_method,
                                     render, render_mode, purpose='bootstrap')
        else:
            # Reset environment
            done = False
            observ, step_reward, done, info = env.reset(start_pose)
            # Start rendering
            if render:
                if env.renderer is None:
                    env.render()
            # Timestep of rollout
            traj = {"observs": [], "poses_x": [], "poses_y": [], "poses_theta": [], "scans": [], "actions": [],
                    "reward": 0}
            for _ in range(max_traj_len):
                # Extract useful observations
                raw_lidar_scan = observ["scans"][0]
                downsampled_scan = agent_utils.downsample_and_extract_lidar(observ, observation_shape,
                                                                            downsampling_method)

                # Add Sim2Real noise
                sim2real_noise = np.random.uniform(-0.25, 0.25, downsampled_scan.shape)
                downsampled_scan = downsampled_scan + sim2real_noise

                linear_vels_x = observ["linear_vels_x"][0]

                poses_x = observ["poses_x"][0]
                poses_y = observ["poses_y"][0]
                poses_theta = observ["poses_theta"][0]
                curr_expert_speed, curr_expert_steer = expert.plan(poses_x, poses_y, poses_theta, tlad, vgain)

                expert_action = np.array([[curr_expert_steer, curr_expert_speed]])
                agent_action_raw = agent.get_action(downsampled_scan)
                agent_action = np.expand_dims(agent_action_raw, axis=0)

                curr_agent_steer = agent_action_raw[0]
                curr_agent_speed = agent_action_raw[1]

                # Decide if agent or expert has control
                if (np.abs(curr_agent_steer - curr_expert_steer) > 0.1) or (
                        np.abs(curr_agent_speed - curr_expert_speed) > 1):
                    """
                    poses_x = observ["poses_x"][0]
                    poses_y = observ["poses_y"][0]
                    poses_theta = observ["poses_theta"][0]

                    curr_expert_speed, curr_expert_steer = expert.plan(poses_x, poses_y, poses_theta, tlad, vgain)
                    curr_action = np.array([[curr_expert_steer, curr_expert_speed]])
                    """
                    curr_action = expert_action

                    traj["observs"].append(observ)
                    traj["scans"].append(downsampled_scan)
                    traj["poses_x"].append(observ["poses_x"][0])
                    traj["poses_y"].append(observ["poses_y"][0])
                    traj["poses_theta"].append(observ["poses_theta"][0])
                    traj["actions"].append(curr_action)
                    traj["reward"] += step_reward
                else:
                    """
                    curr_action_raw = agent.get_action(downsampled_scan)
                    curr_action = np.expand_dims(curr_action_raw, axis=0)
                    """
                    curr_action = agent_action

                observ, reward, done, _ = env.step(curr_action)

                # Update rendering
                if render:
                    env.render(mode=render_mode)

                if done:
                    break

            print("Adding to dataset...")
            if len(traj["observs"]) > 0:
                traj["observs"] = np.vstack(traj["observs"])
                traj["poses_x"] = np.vstack(traj["poses_x"])
                traj["poses_y"] = np.vstack(traj["poses_y"])
                traj["poses_theta"] = np.vstack(traj["poses_theta"])
                traj["scans"] = np.vstack(traj["scans"])
                traj["actions"] = np.vstack(traj["actions"])
                dataset.add(traj)

            log['Number of Samples'].append(dataset.get_num_of_total_samples())
            log['Number of Expert Queries'].append(dataset.get_num_of_total_samples())

            print("Training agent...")
            for _ in range(n_batch_updates_per_iter):
                train_batch = dataset.sample(train_batch_size)
                agent.train(train_batch["scans"], train_batch["actions"])

    agent_utils.save_log_and_model(log, best_model, algo_name)


if __name__ == '__main__':
    # Parse the command line arguments.
    parsed_args = process_parsed_args()

    # Process the parsed arguments.
    il_algo = parsed_args.algorithm
    yaml_loc = parsed_args.training_config

    il_config = yaml.load(open(yaml_loc), Loader=yaml.FullLoader)

    # Initialize
    seed, agent, expert, env, start_pose, observation_shape, downsampling_method, render, render_mode = \
        initialization(il_config)

    # Train
    hg_dagger(seed, agent, expert, env, start_pose, observation_shape, downsampling_method, render, render_mode)
