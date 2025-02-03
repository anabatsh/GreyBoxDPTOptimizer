import numpy as np
import torch
import os
import json
import yaml
import re
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm


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
    i = np.array(i)[::-1].T
    return i

def get_xaxis(d, n, len_max=1024):
    """
    For given d and n list all the vectors [{1, n}]^d
    *if there's too many, list only len_max of them with equal step
    input: d, n - integer
    input: i - either all or a part of the vectors [{1, n}]^d
    """
    if n > 1:
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
    else:
        # x = np.linspace(0, 1, len_max).reshape(-1, 1).repeat(d, axis=1)
        x = np.linspace(-3, 3, len_max).reshape(-1, 1).repeat(d, axis=1)
        return x

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
    x_axis = np.arange(len(y))
    ax.vlines(x_axis[min_index], y_min, y[min_index], colors=color)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('x')

    if problem.n > 1:
        ax.set_xticks([0, len(y)-1], [fr'$[0]^{{{problem.d}}}$', fr'$[{problem.n-1}]^{{{problem.d}}}$'])
    else:
        ax.set_xticks([0, len(y)-1], [fr'$[{i[0][0]}]^{{{problem.d}}}$', fr'$[{i[-1][0]}]^{{{problem.d}}}$'])
    # save_path = os.path.join(save_dir, 'problem.png')
    # plt.savefig(save_path)
    plt.show()

def print_sample(sample, predictions=None, print_ta=True, print_fm=False):
    tab = ' ' * 4

    def state_transform(state):
        x, y = state[:-1].int().tolist(), state[-1].item()
        return f'{str(x)[:-1]}, {y:.6f}]'
    
    def action_transform(action):
        action = action.item()
        return f'{action}'

    def reward_transform(reward):
        reward = reward.item()
        return f'reward: {reward}'

    print('query_state:')
    print(tab, state_transform(sample["query_state"]))

    print('context:')
    if len(sample.keys()) > 3:
        context = zip(
            sample["states"], 
            sample["actions"], 
            sample["next_states"], 
            sample["rewards"]
        )
        if predictions is None:
            for state, action, next_state, reward in context:
                state_str = state_transform(state)
                action_str = action_transform(action)
                next_state_str = state_transform(next_state)
                reward_str = reward_transform(reward)
                print(tab, f'{state_str} -> {action_str} -> {next_state_str} {reward_str}')
        else:
            print(tab, f'{{{action_transform(predictions[0])}}}')
            for prediction, (state, action, next_state, reward) in zip(predictions[1:], context):
                state_str = state_transform(state)
                action_str = action_transform(action)
                next_state_str = state_transform(next_state)
                reward_str = reward_transform(reward)
                print(tab, f'{state_str} -> {action_str} -> {next_state_str} {reward_str}', f'{{{action_transform(prediction)}}}')

    if print_ta:
        if "target_action" in sample:
            print('target_action:')
            print(tab, action_transform(sample["target_action"]))
        elif "target_state" in sample:
            print('target_state:')
            print(tab, state_transform(sample["target_state"]))

    print('ground truth:')
    gt_state = torch.cat([
        torch.tensor(sample["problem"].info["x_min"].copy()),
        torch.tensor([sample["problem"].info["y_min"]])
    ])
    print(tab, state_transform(gt_state))

    if print_fm:
        index_min = np.argmin(sample["next_states"][..., -1])
        print('found minimum:')
        print(tab, state_transform(sample["next_states"][index_min]))

def run(model, sample, n_steps=15):
    if len(sample.keys()) > 3:
        outputs = model.model(
            query_state=sample["query_state"].unsqueeze(0),
            states=sample["states"].unsqueeze(0),
            actions=sample["actions"].unsqueeze(0),
            next_states=sample["next_states"].unsqueeze(0),
            rewards=sample["rewards"].unsqueeze(0)
        )
        predictions = model.get_predictions(outputs)
        targets = sample["target_action"].unsqueeze(0)
        metrics = model.get_loss(outputs, targets, predictions) | model.get_metrics(outputs, targets, predictions)
    else:
        results = model.run(
            query_state=sample["query_state"],
            problem=sample["problem"],
            n_steps=n_steps, #model.config["model_params"]["seq_len"]+1,
            do_sample=model.config["do_sample"],
            temperature_function=lambda x: model.config["temperature"]
        )
        sample["states"] = results["states"]
        sample["actions"] = results["actions"]
        sample["next_states"] = results["next_states"]
        sample["rewards"] = results["rewards"]

        outputs = results["outputs"].unsqueeze(0)
        predictions = results["next_states"].unsqueeze(0)
        targets = sample["target_state"].unsqueeze(0)
        metrics = model.get_metrics(outputs, targets, predictions)

    return sample, outputs.squeeze(0), predictions.squeeze(0), metrics

def print_metrics(metrics):
    tab = ' ' * 4
    print('metrics:')
    for key, val in metrics.items():
        print(f'{tab}{key} = {val.item():.6f}')
