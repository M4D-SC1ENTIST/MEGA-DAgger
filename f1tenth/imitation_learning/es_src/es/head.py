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
Head node of gradient-free optimization. Sets up the optimization and
distributes the workload to different simulation workers.

Author: Hongrui Zheng
Last Modified: 6/27/2022
"""

import os
from argparse import Namespace

import nevergrad as ng
import numpy as np
import ray
from tqdm import tqdm

from es.worker import Worker
from es.utils.DataProcessor import obsDict2carStateSeq, paramsDict2Array


def run_es(conf: Namespace, _run=None):
    """
    Run function that starts optimization.
    """
    # set up filenames and seeding
    # np.random.seed(conf.seed)
    run_id = _run._id

    pkl_path = os.path.abspath(f'../runs/{run_id}/optims_pkl')
    batchData_path = os.path.abspath(f'../runs/{run_id}/batch_data')
    score_path = os.path.abspath(f'../runs/{run_id}/scores')
    artifact_prefix = f'{conf.run_name}_{conf.optim_method}_budget_{str(conf.budget)}'
    os.makedirs(pkl_path, exist_ok=True)
    for i in range(conf.num_workers):
        os.makedirs(os.path.join(batchData_path, f'worker_{i}'))
    os.makedirs(batchData_path, exist_ok=True)
    os.makedirs(score_path, exist_ok=True)

    # setting up parameter space
    param = ng.p.Dict(
        # logarithmically distributed float
        traj_v_scale=ng.p.Scalar(lower=conf.ego_v_lb, upper=conf.ego_v_ub),
        # one-dimensional array of length 2
        cost_weights=ng.p.Array(shape=(conf.num_weights,), lower=conf.weights_lb, upper=conf.weights_ub),
        # collision_thres=ng.p.Scalar(lower=0.2, upper=0.4),
    )
    # create optimization
    optim = ng.optimizers.registry[conf.optim_method](
        parametrization=param, budget=conf.budget, num_workers=conf.num_workers
    )
    # seeding
    optim.parametrization.random_state = np.random.RandomState(conf.seed)

    # setting up workers
    workers = [Worker.remote(conf, conf, worker_id, run_id) for worker_id in range(conf.num_workers)]

    # all scores
    all_scores = []
    all_individuals = []
    all_overtake = []
    all_crash = []

    # create random opponents
    randomGenerator = np.random.default_rng(conf.seed)
    opponent_weights = conf.weights_ub * (randomGenerator.random(size=(conf.num_opp, conf.num_weights+conf.num_params)))
    opponent_weights = opponent_weights.T
    if conf.traj_v_scale:
        opponent_v = (conf.opp_v_ub-conf.opp_v_lb) * randomGenerator.random(size=conf.num_opp) + conf.opp_v_lb
        opponent_weights[0] = opponent_v
    if conf.collision_thres:
        opponent_thres = 0.2 * randomGenerator.random(size=conf.num_opp) + 0.2
        opponent_weights[-1] = opponent_thres
    opponent_weights = opponent_weights.T

    np.savez_compressed(os.path.join(score_path, artifact_prefix + f"_opp_weights.npz"), opp_weights=opponent_weights)

    if conf.resume_train:
        optim = ng.optimizers.base.Optimizer.load(conf.optim_path)
        curr_pareto = optim.pareto_front()
        for i in range(conf.num_opp):
            if i>0:
                raw_weights = curr_pareto.pop().value
                opponent_weights[i, :] = raw_weights
    # else:
    #     for i in range(conf.num_opp):
    #         opponent_weights[i, 0] = np.clip(opponent_weights[i, 0], a_min=6.0, a_max=8.0)

    # opponent_weights = opponent_weights / np.linalg.norm(opponent_weights, ord=1, axis=1).reshape(-1, 1)

    # TODO: tempering schedule
    # conf.mixing_decay

    # work distribution loop
    for epoch in tqdm(range(conf.budget // conf.num_workers)):
        # TODO: tempering scheduling for mixing competitive agents in
        # os.makedirs(os.path.join(batchData_path, f'epoch_{epoch}'), exist_ok=True)
        # if epoch > 0:
        #     # competency update, mix in agents on the pareto front, epsilon-greedy
        #     mixing_percentage = conf.mixing_decay ** epoch
        #     mixing_percentage = np.clip(mixing_percentage, a_min=0.7, a_max=1.0)
        #     try:
        #         curr_pareto = optim.pareto_front()
        #     except:
        #         curr_pareto = []
        #     curr_pareto_size = len(curr_pareto)
        #     counter = 0
        #     # import ipdb; ipdb.set_trace()
        #     # loop through all opponents to decide whether mix in
        #     for i in range(conf.num_opp):
        #         roll = randomGenerator.random(size=1)
        #         if roll >= mixing_percentage and counter < curr_pareto_size:
        #             # if we run out of candidates, stop mixing
        #             raw_weights = curr_pareto.pop().value
        #             # opponent_weights[i, :] = raw_weights / np.linalg.norm(raw_weights, ord=1)
        #             opponent_weights[i, :] = paramsDict2Array(raw_weights, conf.params_name)
        #             counter += 1

        # distribute
        individuals = [optim.ask() for _ in range(conf.num_workers)]
        for ind, worker in zip(individuals, workers):
            # import ipdb; ipdb.set_trace()
            worker.run_sim.remote(ind.value, opponent_weights, epoch, conf.weights_ub, conf.time_per_run)

        # collect
        future_batch_result = [worker.collect.remote() for worker in workers]
        batch_result = ray.get(future_batch_result)  # (worker_num, result_num)
        for i, indvidual_weights, result in zip(range(conf.num_workers), individuals, batch_result):
            score, crash, overtake = result
            optim.tell(indvidual_weights, score)
            all_scores.append(score)
            all_individuals.append(paramsDict2Array(indvidual_weights.value, conf.params_name))
            all_crash.append(crash)
            all_overtake.append(overtake)

        # save
        optim.dump(os.path.join(pkl_path, artifact_prefix + f"epoch{epoch}" + "_optim.pkl"))
        score_all_np = np.asarray(all_scores)
        params_all_np = np.asarray(all_individuals)
        crash_all_np = np.asarray(all_crash)
        overtake_all_np = np.asarray(all_overtake)
        np.savez(os.path.join(score_path, artifact_prefix + f"epoch{epoch}" + "_score.npz"), scores=score_all_np, params=params_all_np, crash=crash_all_np, overtake=overtake_all_np)
