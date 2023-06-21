import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import spatial
import nevergrad as ng
from sklearn.decomposition import PCA

data_module = os.path.abspath('../data')
run = 3
epoch = 95
score_file = f'{run}/scores/default_CMA_budget_9600epoch{epoch}_score.npz'
optimpkl_file = f'{run}/optims_pkl/default_CMA_budget_9600epoch{epoch}_optim.pkl'

objective_dict = {0: 'progress', 1: 'safety'}
show_objective = {0, 1}
worker_num = 100
objective_num = 2

functions = ('show_score', 'show_relation', 'show_pareto')
f_id = 4
font_size = 15


def show_score():

    # objective_dict = {0: 'progress', 1:'aggressive', 2:'safety'}
    data = np.load(os.path.join(data_module, score_file), mmap_mode='r', allow_pickle=True)
    score_series = data['scores'].reshape(-1, worker_num, objective_num)  # (worker*n, objective_num)
    # print(data['params'][-1000:])
    # score_series = data['scores']
    # plt.plot(score_series[:,2], 'o', markersize=1.0, label='progress')


    df = {'iter_step':[], 'objectives':[], 'objective_value':[]}
    for iter_step in range(0, score_series.shape[0]):
        df['iter_step'].extend([iter_step] * worker_num * len(show_objective))
        for i, obj in objective_dict.items():
            if i in show_objective:
                df['objectives'].extend([obj]*worker_num)
                df['objective_value'].extend(list(score_series[iter_step, :, i]))
    df = pd.DataFrame(data=df)
    sns.lineplot(data=df, x='iter_step', y='objective_value', hue='objectives', style='objectives')
    plt.xlabel('iter_step', fontsize=font_size)
    plt.ylabel('objective_value', fontsize=font_size)
    plt.title('Objectives Distribution over Iteration', fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.show()


def show_relation():
    trunc = 5800
    data = np.load(os.path.join(data_module, score_file), mmap_mode='r', allow_pickle=True)
    objective_score = data['scores'].T  # (2, n)
    raw_weights = data['params']
    seeds_num = len(raw_weights)
    weights = [raw_weights[i].value for i in range(seeds_num)]
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(weights)[:trunc, :]

    # data = {'PC_1': pca_results[:, 0], 'PC_2': pca_results[:, 1], 'progress': objective_score[0][:trunc], 'safety': objective_score[1][:trunc]}
    # df = pd.DataFrame(data=data)
    # sns.scatterplot(data=df, x='PC_1', y='PC_2', hue='progress')

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(data['PC_1'], data['PC_2'], data['safety'], marker='o')
    # # ax.scatter(data['PC_1'], data['PC_2'], data['progress'], marker='^')
    #
    # ax.set_xlabel('PC_1', fontsize=font_size)
    # ax.set_ylabel('PC_2', fontsize=font_size)
    # ax.set_zlabel('objective value', fontsize=font_size)
    #
    # plt.title('PCA for prototypes', fontsize=font_size)
    # plt.xlabel('PC_1', fontsize=font_size)
    # plt.ylabel('PC_2', fontsize=font_size)
    objective_score = objective_score.T
    score_dist = []
    weights_cosine_dist = []
    # n = 500
    for i in range(0, 2000):
        for j in range(i+1, 2000):
            score_dist.append(np.linalg.norm(objective_score [i] - objective_score [j]))
            weights_cosine_dist.append(spatial.distance.cosine(weights[i], weights[j]))
            # weights_cosine_dist.append(np.linalg.norm(weights[i] - weights[j]))
    fig, ax = plt.subplots()
    ax.plot(weights_cosine_dist, score_dist, 'o', markersize=0.2)
    plt.title('Distance in planning space(cosine similarity) and in objective space(Euclidean)', fontsize=font_size)
    plt.xlabel('spatial.distance.cosine', fontsize=font_size)
    plt.ylabel('Euclidean distance', fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.show()


def show_pareto():
    optim_path = os.path.join(data_module, optimpkl_file)
    optim = ng.optimizers.base.Optimizer.load(optim_path)
    objective_score = []
    distance = []
    for param in sorted(optim.pareto_front(), key=lambda p: p.losses[0]):
        objective_score.append(param.losses)
        distance.append(np.linalg.norm(param.value))
        raw_ego_weights = param.value
        ego_param = {'traj_v_scale': raw_ego_weights[0] / 8.0, 'cost_weights': raw_ego_weights[1:] * 2}
        print(f"{ego_param} with objectives {param.losses}")
    objective_score = np.array(objective_score).T
    print(f'len of pareto front is {len(distance)}')

    ## draw
    data = {'progress': objective_score[0], 'safety': objective_score[1], 'Euclidean distance': np.round(distance, 1)}
    data_df = pd.DataFrame(data)
    g = sns.scatterplot(data=data_df, x='progress', y='safety', hue='Euclidean distance', legend='brief')
    # plt.legend(title='distance', fontsize=font_size)
    plt.xlabel('progress', fontsize=font_size)
    plt.ylabel('safety', fontsize=font_size)
    plt.title(f'Pareto front with size {len(distance)}', fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.show()


def show_pareto_pca():
    optim_path = os.path.join(data_module, optimpkl_file)
    optim = ng.optimizers.base.Optimizer.load(optim_path)
    objective_score = []
    distance = []
    weights = []
    for param in sorted(optim.pareto_front(), key=lambda p: p.losses[0]):
        objective_score.append(param.losses)
        distance.append(np.linalg.norm(param.value))
        weights.append(param.value)
    objective_score = np.array(objective_score).T
    print(f'len of pareto front is {len(distance)}')

    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(weights)

    ## draw
    data = {'progress': objective_score[0], 'safety': objective_score[1], 'Euclidean distance': np.round(distance, 1),
            'PC_1': pca_results[:, 0], 'PC_2': pca_results[:, 1]}
    data_df = pd.DataFrame(data)
    sns.scatterplot(data=data_df, x='PC_1', y='PC_2', hue='progress')
    # plt.legend(title='distance', fontsize=font_size)
    plt.xlabel('PC_1', fontsize=font_size)
    plt.ylabel('PC_2', fontsize=font_size)
    plt.title(f'Pareto front PCA', fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.show()


def show_pareto_cosine():
    optim_path = os.path.join(data_module, optimpkl_file)
    optim = ng.optimizers.base.Optimizer.load(optim_path)
    objective_score = []
    distance = []
    weights = []
    for param in sorted(optim.pareto_front(), key=lambda p: p.losses[0]):
        objective_score.append(param.losses)
        distance.append(np.linalg.norm(param.value))
        weights.append(param.value)
    objective_score = np.array(objective_score)
    score_dist = []
    weights_cosine_dist = []
    for i in range(0, len(weights)):
        for j in range(i+1, len(weights)):
            score_dist.append(np.linalg.norm(objective_score [i] - objective_score [j]))
            ## weights_cosine_dist.append(spatial.distance.cosine(weights[i], weights[j]))
            weights_cosine_dist.append(np.linalg.norm(weights[i] - weights[j]))
    fig, ax = plt.subplots()
    ax.plot(weights_cosine_dist, score_dist, 'o', markersize=1.5)
    plt.title('Distance in planning space(cosine similarity) and in objective space(Euclidean)', fontsize=font_size)
    plt.xlabel('spatial.distance.cosine', fontsize=font_size)
    plt.ylabel('Euclidean distance', fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.show()


