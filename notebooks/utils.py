import re
import os
import sys
import json
import yaml
import torch
import numpy as np
import lightning as L
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

# from matplotlib import pyplot as plt

root_path = '../'
sys.path.insert(0, root_path)

from problems import Problem
from scripts.create_problem import load_problem_set
from scripts.run_solver import load_results
from train_dpt import DPTSolver, custom_collate_fn, OnlineDataset

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# ----------------------test_data.ipynb------------------------------

def get_Xy(read_dir, problem_list, suffix='test'):
    X = []
    y = []
    for i, problem in enumerate(problem_list):
        problems = load_problem_set(f"{read_dir}/{problem}/{suffix}")
        Q = np.stack([problem.Q.numpy() for problem in problems])
        X_i = Q.reshape(Q.shape[0], -1)
        y_i = np.ones(Q.shape[0]) * i
        X.extend(X_i)
        y.extend(y_i)
    X = np.stack(X)
    y = np.stack(y)
    print(f'X shape: {X.shape}, y shape: {y.shape}')
    return X, y


def get_tsne(X):
    tsne = TSNE(
        n_components=2,      # we want a 2D embedding
        perplexity=min(len(X)-1, 30),       # typical perplexity value; can tune
        learning_rate='auto',# can also specify a numeric value like 200
        init='pca',          # good default initialization
        random_state=42      # for reproducibility
    )
    X_tsne = tsne.fit_transform(X)
    return X_tsne


def show_tsne(problem_list, X, y, title=''):
    plt.figure(figsize=(6, 5))
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='datalim')

    cmap = plt.get_cmap('jet')
    colors = cmap(np.linspace(0, 1, len(problem_list)))
    cmap = mcolors.ListedColormap(colors) # hsv

    bounds = np.arange(len(problem_list) + 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, norm=norm, s=5)
    colorbar = plt.colorbar(scatter, spacing='proportional')
    tick_positions = (bounds[:-1] + bounds[1:]) / 2
    colorbar.set_ticks(tick_positions)
    colorbar.set_ticklabels(problem_list)

    plt.title(title)
    plt.show()


def print_unique(read_dir, problem_list, suffix='test'):
    print('Number of unique x_best values for every problem:')
    for problem_name in problem_list:
        d_x = []
        # d_solver = []
        problems = load_problem_set(f'{read_dir}/{problem_name}/{suffix}')
        for problem in problems:
            d_x.append(problem.info['x_best'])
            # d_solver.append(problem.info['solver'])
        x_unique = torch.unique(torch.stack(d_x), dim=0)
        # solvers, stats = np.unique(d_solver, return_counts=True)
        print(f'{problem_name}: {len(x_unique)}')# {solvers} {stats}')


def get_meta_results(problem, solver, read_dir, suffix='test', budget=100, n_steps=10):
    problem_path = os.path.join(read_dir, problem, suffix)
    problem_results = defaultdict(list)

    m_list = np.linspace(0, budget, n_steps, dtype=np.int32)

    for problem in os.listdir(problem_path):
        solver_path = os.path.join(problem_path, problem, solver)
            
        y_list = []
        for seed in os.listdir(solver_path):
            seed_path = os.path.join(solver_path, seed)
            results = load_results(seed_path)
            y_list.append(np.interp(m_list, results['m_list'], results['y_list']))

        problem_results['y_list (mean)'].append(np.mean(y_list, axis=0))
        problem_results['y_list (std)'].append(np.std(y_list, axis=0))

    problem_results = {
        'm_list': m_list,
        'y_list (mean)': np.mean(problem_results['y_list (mean)'], axis=0),
        'y_list (std)': np.mean(problem_results['y_list (std)'], axis=0),
    }
    return problem_results


def get_trajectory_stats(problem, solver, read_dir, suffix='test'):
    problem_path = os.path.join(read_dir, problem, suffix)
    problem_stats = defaultdict(list)

    for problem in os.listdir(problem_path):
        solver_path = os.path.join(problem_path, problem, solver)
        
        for seed in os.listdir(solver_path):
            seed_path = os.path.join(solver_path, seed)
            results = load_results(seed_path)
            m_list = results['m_list']
            problem_stats['len'].append(len(m_list))
            problem_stats['last'].append(m_list[-1])

    problem_stats = {
        'len (min)': np.min(problem_stats['len']).item(),
        'len (max)': np.max(problem_stats['len']).item(),
        'len (mean)': np.mean(problem_stats['len']).item(),
        'last (min)': np.min(problem_stats['last']).item(),
        'last (max)': np.max(problem_stats['last']).item(),
        'last (mean)': np.mean(problem_stats['last']).item(),
    }
    return problem_stats


def get_problem_averaged_meta_dict(meta_dict, name='all problems'):
    d = {}
    for problem in meta_dict.keys():
        for solver in meta_dict[problem].keys():
            if solver not in d:
                d[solver] = {}
            for key in meta_dict[problem][solver].keys():
                if key not in d[solver]:
                    d[solver][key] = []
                d[solver][key].append(meta_dict[problem][solver][key])
    for solver in d.keys():
        for key in d[solver].keys():
            d[solver][key] = np.mean(d[solver][key], axis=0)
    return {name: d}


def show_meta_results(meta_results):
    problem_list = meta_results.keys()

    m = min(len(problem_list), 4)
    n = int(np.ceil(len(problem_list) / m))

    fig, axes = plt.subplots(n, m, figsize=(4*m, 3*n), gridspec_kw=dict(hspace=0.2, wspace=0.25), sharex=True)
    axes = np.array([axes]) if m == 1 else axes
    axes = axes.reshape(n, m)
    for p, problem in enumerate(problem_list):
        i, j = p // m, p % m

        clip_val = None

        if 'PROTES' in meta_results[problem]:
            clip_val = meta_results[problem]['PROTES']['y_list (mean)'][0]
        else:
            clip_val = None

        for solver in meta_results[problem].keys():
            results = meta_results[problem][solver]
            if 'mode' in problem:
                title = re.findall(r'mode_(.*?)(?=__)', problem)[0]
                if title == 'normal':
                    loc = int(re.findall(r'loc_(.*?)(?=__)', problem)[0])
                    scale = int(re.findall(r'scale_(.*)', problem)[0])
                    title = f'N({loc}, {scale})'
            else:
                title = problem
            axes[i, j].set_title(title)
            y = np.clip(results['y_list (mean)'], None, clip_val)
            axes[i, j].plot(results['m_list'], y, label=solver)
            # axes[i, j].fill_between(
            #     results['m'], 
            #     y - results['y (std)'], 
            #     y + results['y (std)'], 
            #     alpha=0.3
            # )
        axes[i, j].legend(loc=1)
        # axes[i, j].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # axes[i, j].legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, fancybox=True)
    plt.show()

# ----------------------test_model.ipynb-----------------------------

def print_sample(sample, rewards, action_mode='point', context_len_max=10):
    
    def transform_state(state):
        x = state[..., :-1].long()
        y = state[..., -1].float()
        x_str = ''.join([str(i) for i in x.tolist()])
        y_str = f'{y:<8.3f}'
        return x_str + ' ' + y_str
    
    def transform_action(action):
        if action_mode == 'point':
            action_str = ''
        else:
            action_str = action.tolist()
            action_str = ''.join([str(i) for i in action_str])
            # action_str = torch.argmax(action, dim=-1).item()
            # action_str = f'{action_str:<2d}'
        return action_str
    
    def transform_reward(reward):
        reward_str = f'{reward:.3f}'
        return reward_str

    tab = ' ' * 4
    print('context')
    context = list(zip(sample['states'], sample['actions'], sample['next_states'], rewards))
    context_len = min(len(context), context_len_max)
    for i in range(context_len):
        state, action, next_state, reward = context[i]
        state_str = transform_state(state)
        next_state_str = transform_state(next_state)
        action_str = transform_action(action)
        reward_str = transform_reward(reward)
        context_sample_str = f'{tab}{state_str} -> {action_str} -> {next_state_str} -> {reward_str}'
        print(context_sample_str)

    print('query_state')
    quer_state_str = transform_state(sample['query_state'])
    quer_state_str = f'{tab}{quer_state_str}'
    print(quer_state_str)

    print('target_action')
    target_action_str = transform_action(sample['target_action'])
    target_action_str = f'{tab}{target_action_str}'
    print(target_action_str)

    print('target_state')
    target_state_str = transform_state(sample['target_state'])
    target_state_str = f'{tab}{target_state_str}'
    print(target_state_str)

# ----------------------test_model.ipynb-----------------------------

def run_model(model, read_dir, problem, name, suffix='test', budget=100):
    problem_path = os.path.join(read_dir, problem, suffix)
    problems = load_problem_set(problem_path)
    dataset = OnlineDataset(problems)
    collate_fn = partial(custom_collate_fn, problem_class=Problem)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1000,
        # num_workers=1,
        # pin_memory=True,
        # shuffle=False,
        collate_fn=collate_fn
    )
    tester = L.Trainer(logger=False, precision=model.config['precision'])
    logs = {}
    for warmup, do_sample in ((0, False), (0, True), (50, False), (50, True)):
        model.config['do_sample'] = do_sample
        model.config['warmup_steps'] = warmup
        model.config['online_steps'] = int(budget - warmup)

        with torch.inference_mode():
            tester.test(model=model, dataloaders=dataloader)

        warmup_mode = 'warmup' if warmup > 0 else 'no warmup'
        sample_mode = 'sample' if do_sample else 'argmax'
        logs[f'{name} ({warmup_mode}) ({sample_mode})'] = {
            'm_list': np.arange(budget) + 1,
            'y_list (mean)': model.trajectory.cpu().numpy()
        }
    return logs

# -------------------------------------------------------------------