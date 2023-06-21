import nevergrad as ng
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


def multiobjective(x):
    return [np.sum(x**2), np.sum((x - 1) ** 2)]


def optimize():
    optim_path = './optim.pkl'
    optim_path = os.path.abspath(optim_path)
    print(os.path.exists(optim_path))

    parametrization = ng.p.Array(shape=(2,), lower=0.0, upper=1.0)
    budget = 1000
    num_workers = 50
    optim = ng.optimizers.registry['CMA']\
        (parametrization=parametrization, budget=budget, num_workers=num_workers)
    optim.parametrization.random_state = np.random.RandomState(6300)

    for _ in range(budget // num_workers):
        x = []
        y = []
        for i in range(num_workers):
            x.append(optim.ask())
            y.append(multiobjective(x[-1].value))
        for i in range(num_workers):
            optim.tell(x[i], y[i])
    objective_score = []
    distance = []
    # for param in sorted(optim.pareto_front(), key=lambda p: p.losses[0]):
    for param in optim.pareto_front():
        objective_score.append(param.losses)
        distance.append(np.linalg.norm(param.value))
        print(f"{param} with losses {param.losses}")
    objective_score = np.array(objective_score).T
    # print("* ", 'CMA', " provides a vector of parameters with test error ",
    #       multiobjective(*recommendation.args, **recommendation.kwargs))
    # optim.dump(optim_path)
    print(f'one ask example {optim.ask()}')
    print('save optim pkl')

    ## draw

    fig, ax = plt.subplots(figsize=(20, 10))
    sns.scatterplot(x=objective_score[0], y=objective_score[1], hue=distance)
    plt.show()


def load_optim():
    optim_path = './optim.pkl'
    parametrization = ng.p.Array(shape=(4,))
    budget = 300
    num_workers = 3
    # optim = ng.optimizers.registry['CMA']\
    #     (parametrization=parametrization, budget=budget, num_workers=num_workers)
    # optim.load(optim_path)
    # optim=ng.optimizers.base.Optimizer.load(optim_path)
    # print(f'one ask example {optim.ask().value}')
    # for param in sorted(optim.pareto_front(), key=lambda p: p.losses[0])[:3]:
    #     print(f"{param} with losses {param.losses}")


optimize()
# load_optim()
