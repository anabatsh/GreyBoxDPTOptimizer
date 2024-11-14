# from solvers.dpt.utils.data import results2trajectories
import os
import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import WandbLogger

from solvers.dpt.train import load_config, DPTSolver
from solvers.dpt.utils.data import MarkovianDataset, MarkovianOfflineDataset

from problems import Net
from utils import *

os.environ["WANDB_SILENT"] = "true"


def get_query_state_and_target_action(problem):
    all_actions = get_xaxis(d=problem.d, n=problem.n)
    all_states = problem.target(all_actions)
    query_state = torch.tensor([all_states.max()])
    i = np.argmin(all_states)
    target_action = all_actions[i]
    base = problem.n ** np.arange(problem.d)[::-1]
    target_action = torch.tensor(all_actions[i] @ base)
    return query_state, target_action

def get_query_states_and_target_actions(problems):
    query_states = []
    target_actions = []
    for problem in problems:
        query_state, target_action = get_query_state_and_target_action(problem)
        query_states.append(query_state)
        target_actions.append(target_action)
    return query_states, target_actions

def transition_function(state, action, problem):
    point = int2bin(action, d=problem.d, n=problem.n)
    target = problem.target(point)
    state = torch.tensor(target).reshape(-1, 1)
    return state

def reward_function(states, actions, next_states, problem):
    return -1 * (next_states - states)[:, -1, 0]

def get_offline_dataloaders(config):
    train_offline_dataset = MarkovianDataset(config["train_histories_path"], seq_len=config["model_params"]["seq_len"])
    train_offline_dataloader = DataLoader(dataset=train_offline_dataset, batch_size=config["batch_size"])
    print('Train dataset:', len(train_offline_dataset))

    val_offline_dataset = MarkovianDataset(config["test_histories_path"], seq_len=config["model_params"]["seq_len"])
    val_offline_dataloader = DataLoader(dataset=val_offline_dataset, batch_size=config["batch_size"])
    print('Validation dataset:', len(val_offline_dataset))

    # test_offline_dataset = MarkovianDataset("../GreyBoxDPTOptimizerData/trajectories_test/", seq_len=config["model_params"]["seq_len"])
    # test_offline_dataloader = DataLoader(dataset=test_offline_dataset, batch_size=config["batch_size"])
    # print('Test dataset:', len(test_offline_dataset))
    return train_offline_dataloader, val_offline_dataloader

def get_online_dataloader(config, problems):
    query_states, target_actions = get_query_states_and_target_actions(problems)
    dataset = MarkovianOfflineDataset(
        problems, query_states, transition_function, 
        reward_function, target_actions, 
        seq_len=config["model_params"]["seq_len"]
    )
    dataloader = DataLoader(dataset=dataset, batch_size=1, collate_fn=lambda batch: batch)
    print('Val dataset:', len(dataset))
    return dataloader

def train(config):
    # results2trajectories(
    #     read_dir='../GreyBoxDPTOptimizerData/results_test', 
    #     save_dir='../GreyBoxDPTOptimizerData/trajectories_test', 
    # )

    train_offline_dataloader, val_offline_dataloader = get_offline_dataloaders(config)

    d = int(np.log2(config["model_params"]["num_actions"]))
    val_online_dataloader = get_online_dataloader(config, [Net(d=d, n=2, seed=i) for i in range(271, 281)])
    val_train_online_dataloader = get_online_dataloader(config, [Net(d=d, n=2, seed=i) for i in range(1, 11)])

    logger = WandbLogger(**config["wandb_params"])

    model = DPTSolver('config.yaml')

    trainer = L.Trainer(
        precision='16-mixed',
        logger=logger,
        max_epochs=config["max_epochs"],
        log_every_n_steps=config["log_every_n_steps"],
        default_root_dir=config["wandb_params"]["save_dir"],
        enable_model_summary=False,
    )
    logger.experiment.config.update({"save_dir": trainer.default_root_dir})

    trainer.fit(
        model=model, 
        train_dataloaders=train_offline_dataloader, 
        val_dataloaders=[val_offline_dataloader, val_online_dataloader, val_train_online_dataloader]
    )
    # trainer.test(model, dataloaders=test_dataloader)

if __name__ == '__main__':
    config = load_config('config.yaml')['TrainConfig']
    train(config)