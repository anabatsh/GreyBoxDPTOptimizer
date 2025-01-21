import numpy as np
import torch
import os
import json
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
from prettytable import PrettyTable
from argparse import ArgumentParser
import problems


def int2bin(x, d, n):
    """
    For a given decimal scalar value obtain a 
    corresponding vector of length d in base-n system
    input: x - batch size decimal scalars in [0, n^d)
           d, n - integer
    input: i - batch size vectors [{1, n}]^d
    """
    i = []
    for _ in range(d):
        i.append(x % n)
        x = x // n
    if isinstance(x, torch.Tensor):
        i = torch.stack(i).T.flip(-1)
    else:
        i = np.array(i)[::-1].T
    return i

def get_xaxis(d, n, len_max=1024):
    """
    For given d and n list all the vectors [{1, n}]^d
    *if there's too many, list only len_max of them with equal step
    input: d, n - integer
    input: i - either all or a part of the vectors [{1, n}]^d
    """
    if d == 1:
        i = np.linspace(0, n-1, len_max).astype(np.int32).reshape(-1, 1)
        return i
    d_new = min(d, int(np.emath.logn(n, len_max)))
    x = np.arange(0, n ** d_new)
    i = int2bin(x, d_new, n)
    if d_new < d:
        i = np.pad(i, ((0, 0), (0, d - d_new)), constant_values=0)
        i = np.pad(i, ((0, 1), (0, 0)), constant_values=n-1)
    return i

def show_problem(problem, save_dir='', ax=None, color='red', linestyle='-o', x_min=True):
    """
    Vizualize a given problem as a 2D plot and save it. 
    For all (or some reasonable amount) of the points from the problem argument space
    compute the corresponding target values and depict them on a 2D plot, where
    - axis x corresponds to the decimal representation of the argument vectors,
    - axis y corresponds to the target values,
    and the points that don't satisfy the constraints are punctured
    input: problem
           save_dir - directory to save the result
    """
    i = get_xaxis(problem.d, problem.n)
    y = problem.target(i)

    if ax == None:
        plt.figure(figsize=(8, 4))
        ax = plt.gca()

    ax.set_title('Target Function')

    ax.plot(y, linestyle, c=color, markersize=1)

    min_index = np.argmin(y)
    y_min, y_max = ax.get_ylim()
    ax.vlines(i[min_index], y_min, y[min_index], colors=color)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('x')

    # ax.set_xticks([0, len(y)-1], [fr'$[0]^{{{problem.d}}}$', fr'$[{problem.n-1}]^{{{problem.d}}}$'])
    # save_path = os.path.join(save_dir, 'problem.png')
    # plt.savefig(save_path)
    # plt.show()