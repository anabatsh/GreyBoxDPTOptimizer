import os
import yaml
import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import seed_everything

from solvers.dpt.train import DPTSolver
from solvers.dpt.utils.data import MarkovianOfflineDataset, MarkovianOnlineDataset

from utils import *
import problems as p

os.environ['WANDB_SILENT'] = "true"


def results2trajectories(
        read_dir='results', 
        save_dir='trajectories', 
        solvers=['NoisyBandit', 'PSO', 'OnePlusOne', 'Portfolio', 'PSO', 'SPSA', 'RandomSearch']
    ):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    for problem in os.listdir(read_dir):
        problem_dir = os.path.join(read_dir, problem)
        solvers = solvers if len(solvers) else os.listdir(problem_dir)
        for solver in solvers:
            solver_dir = os.path.join(problem_dir, solver)
            if os.path.isdir(solver_dir):
                for seed in os.listdir(solver_dir):
                    seed_dir = os.path.join(solver_dir, seed)
                    with open(os.path.join(seed_dir, 'logs.txt')) as f:
                        r = f.read().split('\n')[:-4][1::2]
                        arguments = []
                        targets = []
                        constraints = []
                        for row in r:
                            row_argument_target, row_constraint = row.split('|')
                            row_argument, row_target = row_argument_target.split('->')
                            row_argument = list(row_argument.strip()[1:-1].split(','))

                            argument_value = np.array([int(x) for x in row_argument])
                            base_2 = 2 ** np.arange(len(argument_value))[::-1]
                            argument_value = argument_value @ base_2
                            arguments.append(argument_value)

                            target_value = float(row_target)
                            targets.append(target_value)
                            
                            constraint_value = bool(row_constraint)
                            constraints.append(constraint_value)
                            
                    actions = np.array(arguments)
                    states = np.array(targets)
                    ground_truth = actions[np.argmin(states)]

                    states = np.hstack([np.zeros(1), states])
                    history = {
                        "states": states.reshape(-1, 1),
                        "actions": actions,
                        "rewards": -1 * (states[1:] - states[:-1]),
                        "target_actions": np.array([ground_truth] * len(r))
                    }
                    np.savez(f'{save_dir}/{problem}_{solver}_{seed}', **history, allow_pickle=True)

def problems2trajectories(
        problems=[],
        save_dir='trajectories', 
    ):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    for problem in problems:
        base = problem.n ** np.arange(problem.d)[::-1]
        all_actions = get_xaxis(d=problem.d, n=problem.n)
        all_states = problem.target(all_actions)
        all_actions = all_actions @ base
        target_action = all_actions[np.argmin(all_states)]
        states = np.hstack([np.zeros(1), all_states])
        history = {
            "states": states.reshape(-1, 1),
            "actions": all_actions,
            "rewards": -1 * (states[1:] - states[:-1]),
            "target_actions": np.array([target_action] * len(all_actions))
        }
        np.savez(f'{save_dir}/{problem.name}', **history, allow_pickle=True)

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Config file not found: {config_path}')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_query_state_and_target_action(problem, config):
    all_actions = get_xaxis(d=problem.d, n=problem.n)
    all_states = problem.target(all_actions)
    i = np.argmin(all_states)
    target_action = all_actions[i]
    base = problem.n ** np.arange(problem.d)[::-1]
    target_action = all_actions[i] @ base
    query_state = np.array([all_states.max()])
    return query_state, target_action

def get_query_states_and_target_actions(problems, config):
    query_states = []
    target_actions = []
    for problem in problems:
        query_state, target_action = get_query_state_and_target_action(problem, config)
        query_states.append(query_state)
        target_actions.append(target_action)
    return query_states, target_actions

def transition_function(state, action, problem):
    point = int2bin(action, d=problem.d, n=problem.n)
    target = problem.target(point)
    state = torch.tensor(target).reshape(-1, 1)
    return state

def reward_function(states, actions, next_states, problem):
    return (states - next_states)[:, -1, 0]

def get_dataloaders(config):
    # get the problems
    problem_class = getattr(p, config["problem"])
    problems = [problem_class(**config["problem_params"], seed=i) for i in range(config["n_problems"])]

    # get trajectories if there is none
    if not os.path.exists(config["trajectories_path"]):
        if "results_path" in config and os.path.exists(config["results_path"]):
            results2trajectories(read_dir=config["results_path"], save_dir=config["trajectories_path"])
        else:
            problems2trajectories(problems=problems, save_dir=config["trajectories_path"])

    # get an offline train and validation dataloaders
    offline_dataset = MarkovianOfflineDataset(config["trajectories_path"], seq_len=config["model_params"]["seq_len"], ordered=config["ordered"])
    train_offline_dataset, val_offline_dataset = torch.utils.data.random_split(offline_dataset, [0.8, 0.2])
    train_offline_dataloader = DataLoader(
        dataset=train_offline_dataset, batch_size=config["batch_size"], 
        shuffle=True, num_workers=config["num_workers"], pin_memory=True
    )
    val_offline_dataloader = DataLoader(
        dataset=val_offline_dataset, batch_size=config["batch_size"], 
        shuffle=True, num_workers=config["num_workers"], pin_memory=True
    )
    print('train_offline_dataset:', len(train_offline_dataset))
    print('val_offline_dataset:', len(val_offline_dataset))

    # # get an online validation dataloader
    # val_problems = [problems[i] for i in val_offline_dataset.indices]
    # query_states, target_actions = get_query_states_and_target_actions(val_problems, config)
    # val_online_dataset = MarkovianOnlineDataset(val_problems, query_states, transition_function, reward_function, target_actions)
    # val_online_dataloader = DataLoader(dataset=val_online_dataset, batch_size=1, num_workers=config["num_workers"], collate_fn=lambda batch: batch)
    # print('val_online_dataset:', len(val_online_dataset))

    return {
        'train_dataloaders': train_offline_dataloader, 
        'val_dataloaders': [val_offline_dataloader]#, val_online_dataloader]
    }

def train(config):
    logger = WandbLogger(**config["wandb_params"])
    model = DPTSolver(config)
    
    trainer = L.Trainer(
        logger=logger,
        precision=config["precision"],
        max_epochs=config["max_epochs"],
        log_every_n_steps=config["log_every_n_steps"],
        default_root_dir=config["wandb_params"]["save_dir"],
        enable_model_summary=False
        # deterministic=True
    )
    trainer.fit(model=model, **get_dataloaders(config))
    # trainer.test(model, dataloaders=test_dataloader)

if __name__ == '__main__':
    config = load_config('dpt_run_config.yaml')
    # seed_everything(config["seed"], workers=True)
    train(config)