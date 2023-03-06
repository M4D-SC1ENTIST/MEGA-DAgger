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
Author: Hongrui Zheng
Last Modified: 9/3/2022
"""

import argparse

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from dppy.finite_dpps import FiniteDPP
from scipy.spatial.distance import pdist, squareform

parser = argparse.ArgumentParser()
parser.add_argument("--near_pareto_path", type=str, required=True)
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--sample_num", type=int, required=True)
parser.add_argument("--seed", type=int, default=6300)
args = parser.parse_args()

# seeding
rng = np.random.RandomState(args.seed)
np.random.seed(args.seed)

near_pareto = np.load(args.near_pareto_path)
data = np.load(args.data_path)

near_pareto_idx = near_pareto["near_idx"]
obj = data["scores"]
cost_weights = data["params"]
near_pareto_obj = obj[near_pareto_idx]
print(len(near_pareto_idx))

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

# plotting
sns.set_style("white")
sns.set_style("ticks")
sns.set_context("poster")
plt.scatter(near_pareto_obj[:, 0], near_pareto_obj[:, 1], c="b", alpha=0.3, label='All Samples')
plt.scatter(final_samples[:, 0], final_samples[:, 1], s=45, c="r", alpha=1.0, label='DPP Samples')
plt.legend()
plt.xlabel('Aggressiveness')
plt.ylabel('Restraint')
plt.show()
