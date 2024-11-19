# from solvers.dpt.utils.data import results2trajectories
import os
import yaml
import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import seed_everything

from solvers.dpt.train import DPTSolver
from solvers.dpt.utils.data import MarkovianDataset, MarkovianOfflineDataset

from problems import Net
from utils import *

os.environ['WANDB_SILENT'] = "true"


def results2trajectories(
        read_dir='results', 
        save_dir='trajectories', 
        solvers=['NoisyBandit', 'PSO', 'OnePlusOne', 'Portfolio', 'PSO', 'SPSA', 'RandomSearch']
    ):

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

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

                    n = len(r)
                    history = {
                        "states": np.roll(states, 1).reshape(-1, 1),
                        "actions": actions,
                        "reward": (states - np.roll(states, 1))[:, 0],
                        "target_actions": np.array([ground_truth] * n)
                    }
                    np.savez(f'{save_dir}/{problem}_{solver}_{seed}', **history, allow_pickle=True)

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

def get_offline_dataloader(histories_path, config):
    dataset = MarkovianDataset(histories_path, seq_len=config["model_params"]["seq_len"])
    dataloader = DataLoader(dataset=dataset, batch_size=config["batch_size"], num_workers=config["num_workers"])
    return dataloader

def get_online_dataloader(problems, config):
    query_states, target_actions = get_query_states_and_target_actions(problems, config)
    dataset = MarkovianOfflineDataset(
        problems, query_states, transition_function, 
        reward_function, target_actions
    )
    dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=config["num_workers"], collate_fn=lambda batch: batch)
    return dataloader

def train(config):
    # results2trajectories(
    #     read_dir='../GreyBoxDPTOptimizerData/results_test', 
    #     save_dir='../GreyBoxDPTOptimizerData/trajectories_test', 
    # )

    train_offline_dataloader = get_offline_dataloader(config["train_histories_path"], config)
    val_offline_dataloader = get_offline_dataloader(config["val_histories_path"], config)
    print('train_offline_dataloader:', len(train_offline_dataloader.dataset))
    print('val_offline_dataloader:', len(val_offline_dataloader.dataset))

    d = int(np.log2(config["model_params"]["num_actions"]))
    val_online_dataloader_1 = get_online_dataloader([Net(d=d, n=2, seed=i) for i in range(271, 281)], config)
    val_online_dataloader_2 = get_online_dataloader([Net(d=d, n=2, seed=i) for i in range(1, 11)], config)
    print('val_online_dataloader_1:', len(val_online_dataloader_1.dataset))
    print('val_online_dataloader_2:', len(val_online_dataloader_2.dataset))

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
    trainer.fit(
        model=model, 
        train_dataloaders=train_offline_dataloader, 
        val_dataloaders=[val_offline_dataloader, val_online_dataloader_1, val_online_dataloader_2]
    )
    # trainer.test(model, dataloaders=test_dataloader)

if __name__ == '__main__':
    config = load_config('config.yaml')
    seed_everything(config["seed"], workers=True)
    train(config)