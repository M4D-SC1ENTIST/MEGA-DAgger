import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import spatial
import nevergrad as ng
from sklearn.decomposition import PCA
import json
from es.utils.visualize import label_point, draw_pts
from es.utils.DataProcessor import paramsDict2Array


sns.set_style('white')
sns.set_style('ticks')
sns.set_context('poster')

########################### LOAD MODEL ############################
# data_module = os.path.abspath('../data')
store_dir = 'data'
# store_dir = 'es_model'
inside = False
if inside:
    data_module = os.path.abspath('..')
else:
    data_module = os.path.abspath(f'../{store_dir}')

run = 26
epoch = 120
budget = 19200
worker_num = 100
objective_num = 2
scene_num = 120

if store_dir == 'es_model':
    score_file = f'{run}/default_CMA_budget_{budget}epoch{epoch}_score.npz'
    optimpkl_file = f'{run}/default_CMA_budget_{budget}epoch{epoch}_optim.pkl'
    oppo_file = f'{run}/default_CMA_budget_{budget}_opp_weights.npz'
else:
    score_file = f'{run}/scores/default_CMA_budget_{budget}epoch{epoch}_score.npz'
    optimpkl_file = f'{run}/optims_pkl/default_CMA_budget_{budget}epoch{epoch}_optim.pkl'
    oppo_file = f'{run}/scores/default_CMA_budget_{budget}_opp_weights.npz'
scenario_file = f'{run}/rollout_scenarios.npz'
config_file = f'{run}/config.json'

font_size = 30

score_data = np.load(os.path.join(data_module, score_file), mmap_mode='r', allow_pickle=True)
all_scores = score_data['scores']
params_obj = score_data['params']
config_data = json.load(open(os.path.join(data_module, config_file)))
oppo_data = np.load(os.path.join(data_module, oppo_file), mmap_mode='r')
try:
    scenario_data = np.load(os.path.join(data_module, scenario_file), mmap_mode='r', allow_pickle=True)
except:
    pass
opp_weights = oppo_data['opp_weights']

# all_seeds = []
# for i in range(len(scores)):
#     all_seeds.append(params_obj[i].value)
all_seeds = params_obj
all_seeds = np.array(all_seeds).T
all_scores = all_scores.T

try:
    param_dict = config_data['params_dict']
    param_name_list = config_data['params_name']
    print('load param dict from config')
except:
    pass
print(param_name_list)
optim_path = os.path.join(data_module, optimpkl_file)
optim = ng.optimizers.base.Optimizer.load(optim_path)

# objective_dict = {0: 'Aggressiveness', 1:'Restraint'}
objective_dict = {0: 'progress', 1:'safety'}
show_objective = {0, 1}
seeds_w_score = {}
for object_idx in show_objective:
    seeds_w_score[objective_dict[object_idx]] = all_scores[object_idx]
for param_idx, param_name in param_dict.items():
    seeds_w_score[param_name] = all_seeds[int(param_idx)]

seeds_w_score_df = pd.DataFrame(seeds_w_score)

################## ALL SEED ###################
objective_score = []
distance = []
pareto_arr = []
pareto_size = 100000
for i in np.argsort(all_scores.T[:, 0]):
    pareto_size -= 1
    objective_score.append(all_scores.T[i])
    pareto_arr.append(all_seeds.T[i])
    if pareto_size < 0:
        break

pareto_arr = np.array(pareto_arr)
pareto_arr = pareto_arr.T
objective_score = np.array(objective_score).T
print(f'len of pareto front is {len(pareto_arr.T)}')
## draw
data = {'progress': objective_score[0], 'safety': objective_score[1]}
data_df = pd.DataFrame(data)


################## PARETO FRONT ###################
objective_score = []
distance = []
pareto_arr = []
for param in sorted(optim.pareto_front(), key=lambda p: p.losses[0]):
    objective_score.append(param.losses)
    param_arr = paramsDict2Array(param.value, param_name_list)
    pareto_arr.append(param_arr)
    distance.append(np.linalg.norm(param_arr))


pareto_arr = np.array(pareto_arr)
pareto_arr = pareto_arr.T
objective_score = np.array(objective_score).T
print(f'len of pareto front is {len(pareto_arr.T)}')
data = {'progress': objective_score[0], 'safety': objective_score[1]}
data_df_pf = pd.DataFrame(data)


################## NEAR PARETO FRONT ###################
dist_thres = 0.3
near_idx = []
near_dict = {}
for i, pareto_p in enumerate(objective_score.T):
    # (2, )
    dist = np.linalg.norm(all_scores.T - pareto_p, axis=1)  # (9600, )
    if i > objective_score.shape[1] / 3:
        tmp = np.nonzero(dist<dist_thres/3)[0];
    elif i > objective_score.shape[1] *2 / 3:
        tmp = np.nonzero(dist<dist_thres/3)[0];
    else:
        tmp = np.nonzero(dist<dist_thres)[0];
    near_idx.extend(tmp)
    near_dict[i] = []
    near_dict[i].append(tmp)

unique_near_idx_set = set()
unique_near_idx = []
for idx in near_idx:
    if idx not in unique_near_idx_set:
        unique_near_idx_set.add(idx)
        unique_near_idx.append(idx)
near_idx = np.array(unique_near_idx)
# np.savez(os.path.join(data_module, str(run), 'near_pareto_idx.npz'), near_idx=unique_near_idx)
print(os.path.join(data_module, str(run), 'near_pareto_idx.npz'))
# near_idx = np.unique(near_idx)


near_pareto_seeds = all_seeds.T[near_idx]  # (n, 9)
near_pareto_scores = all_scores.T[near_idx]  # (n, 2)

distance = []
for p, s in zip(near_pareto_seeds, near_pareto_scores):
    distance.append(np.linalg.norm(p))
print(f'near pareto front dist threshold is {dist_thres}, near pareto front point number is {len(near_pareto_seeds)}')
# print(distance)
cluster_num = len(near_dict.keys())
cluster_df = pd.DataFrame(columns=seeds_w_score_df.columns.to_list()+['cluster_id'])
for i in near_dict.keys():
    near_idx = near_dict[i][0]
    aug = seeds_w_score_df.iloc[near_idx]
    aug = aug.assign(cluster_id = [i] * len(near_idx))
    # aug['cluster_id'] = [i] * len(near_idx)
    cluster_df = pd.concat((cluster_df, aug), ignore_index=True)


############################## DPP #################################
from argparse import Namespace
from scipy.spatial.distance import pdist, squareform
from dppy.finite_dpps import FiniteDPP
args = {}
args["near_pareto_path"] = f'../data/{run}/near_pareto_idx.npz'
args["data_path"] = os.path.join(data_module, score_file)
args["sample_num"] = 20
args["seed"] = 6300
args = Namespace(**args)

rng = np.random.RandomState(args.seed)
np.random.seed(args.seed)

near_pareto = np.load(args.near_pareto_path)
data = np.load(args.data_path)

near_pareto_idx = near_pareto["near_idx"]
obj = data["scores"]
cost_weights = data["params"]
near_pareto_obj = obj[near_pareto_idx]

# likelihood kernel
l = 1
mcmc_iter = 30
L = np.exp(-(1 / (2 * (l**2))) * squareform(pdist(near_pareto_obj, "sqeuclidean")))
# k-DPP
DPP = FiniteDPP("likelihood", **{"L": L})
# sample
DPP.flush_samples()
for mcmc_iter in range(mcmc_iter):
    DPP.sample_exact_k_dpp(size=args.sample_num, random_state=rng)

final_sample_idx = DPP.list_of_samples[-1]
final_samples = obj[near_pareto_idx[final_sample_idx]]
print("Final sample indices", near_pareto_idx[final_sample_idx])

fig = plt.figure()

gs = fig.add_gridspec(1, 4, hspace=0, wspace=0)
ax = gs.subplots(sharex=True, sharey=True)

ax[0].scatter(data_df['progress'], data_df['safety'], s=60, edgecolors='w', linewidths=0.1)
# sns.scatterplot(data=data_df, x='progress', y='safety', legend='brief', s=60, ax=ax[0],linewidth=0.1)
ax[0].set_ylabel('Restraint', labelpad=18)
ax[0].set_xlabel('Aggressiveness', labelpad=18)
ax[0].set_title('All Agents')
# ax[0].legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)

sns.scatterplot(data=data_df, x='progress', y='safety', legend='brief', alpha=0.1, s=60, ax=ax[1],linewidth=0.1)
# sns.scatterplot(data=data_df_pf, x='progress', y='safety', label='Pareto front', s=60, palette='tab10', ax=ax[1])
sns.scatterplot(data=data_df_pf, x='progress', y='safety', s=120, palette='tab10', ax=ax[1],linewidth=0.2)
# plt.legend(loc=2, prop={'size': 6})
ax[1].set_xlabel('', labelpad=None)
ax[1].set_title('Pareto Front(Optimal Set)')
# ax[1].set_yticks([])
# ax[1][0].set_ylabel('Manuver Rates')
# ax[1][0].set_title('Crash and Overtake Rates')
# ax[1][0].legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)


sns.scatterplot(data=cluster_df, x='progress', y='safety', hue='cluster_id', linewidth=0.2, legend=False, s=120, palette="tab10", ax=ax[2])
ax[2].set_xlabel('', labelpad=None)
ax[2].set_title('Near-optimal Set')
# ax[0][1].set_ylabel('Objective Scores', labelpad=25)
# ax[0][0].set_title('Objectives Scores')
# ax[2].legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)

sns.scatterplot(data=cluster_df, x='progress', y='safety', legend=False, s=120, alpha=0.1, ax=ax[3],linewidth=0.1)
# sns.scatterplot(x=final_samples[:, 0], y=final_samples[:, 1], s=60, label='DPP samples',  ax=ax[3])
sns.scatterplot(x=final_samples[:, 0], y=final_samples[:, 1], s=120, ax=ax[3],linewidth=0.2)
ax[3].set_xlabel('', labelpad=None)
ax[3].set_title('DPP Samples')

# fig.title('title', y=-0.2)
plt.margins(0,0)
plt.savefig('pf_glob_4.eps', format='eps')
plt.savefig('pf_glob_4.png', format='png')

plt.show()