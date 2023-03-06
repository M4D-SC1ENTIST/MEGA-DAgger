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
Experiment orchestration with sacred

Author: Hongrui Zheng
Last Modified: 6/22/2022
"""

from argparse import Namespace

import ray
from sacred import Experiment
from sacred.observers import FileStorageObserver

from es.head import run_es

# setting up sacred experiment
ex = Experiment('obj_space_planning')
ex.observers.append(FileStorageObserver('../runs'))


# tune kinematic simulation
@ex.named_config
def default():
    ex.add_config('../configs/default.yaml')
    ex.add_config('../configs/General1/lattice_planner_config.yaml')


# main
@ex.automain
def run(_run, _config):
    ray.init()
    conf = Namespace(**_config)
#     import ipdb; ipdb.set_trace()
    print(conf)
    run_es(conf, _run)

