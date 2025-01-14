import os
import yaml
import random
from functools import partial
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import collate, default_collate_fn_map

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import seed_everything

from solvers.dpt.train import DPTSolver
from solvers.dpt.utils.data import MarkovianOfflineDataset, MarkovianOnlineDataset

from utils import *
import problems as p

os.environ['WANDB_SILENT'] = "true"

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Config file not found: {config_path}')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def collate_problem_fn(batch, *, collate_fn_map):
    return batch
    
def custom_collate_fn(batch, problem_class):
    custom_collate_fn_map = default_collate_fn_map | {problem_class: collate_problem_fn}
    return collate(batch, collate_fn_map=custom_collate_fn_map)

def get_dataloaders(config):
    # get problems
    problem_class = getattr(p, config["problem"])
    problems = [problem_class(**config["problem_params"], seed=i) for i in range(config["n_problems"])]
    random.shuffle(problems)

    train_problems = problems[:-100]
    val_problems = problems[-100:]

    collate_fn = partial(custom_collate_fn, problem_class=problem_class)
    # get an offline train dataloader
    train_offline_dataset = MarkovianOfflineDataset(
        problems=train_problems,
        seq_len=config["model_params"]["seq_len"],
        ordered=config["ordered"],
        remove_target=config["remove_target"]
    )
    train_offline_dataloader = DataLoader(
        dataset=train_offline_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        pin_memory=True,
        shuffle=True,
        collate_fn=collate_fn
    )
    print('train_offline_dataset:', len(train_offline_dataset))

    # get an offline validation dataloader
    val_offline_dataset = MarkovianOfflineDataset(
        problems=val_problems,
        seq_len=config["model_params"]["seq_len"],
        ordered=config["ordered"],
        remove_target=config["remove_target"]
    )
    val_offline_dataloader = DataLoader(
        dataset=val_offline_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        pin_memory=True,
        shuffle=False,
        collate_fn=collate_fn
    )
    print('val_offline_dataset:', len(val_offline_dataset))

    # get an online validation dataloader
    val_online_dataset = MarkovianOnlineDataset(val_problems)
    val_online_dataloader = DataLoader(
        dataset=val_online_dataset,
        batch_size=1,
        num_workers=config["num_workers"],
        collate_fn=collate_fn
    )
    print('val_online_dataset:', len(val_online_dataset))

    return {
        'train_dataloaders': train_offline_dataloader,
        'val_dataloaders': [val_offline_dataloader, val_online_dataloader]
    }

def train(config):
    logger = WandbLogger(**config["wandb_params"])
    model = DPTSolver(config)

    # checkpoint_callback = ModelCheckpoint(every_n_epochs=100, save_top_k=-1)    
    trainer = L.Trainer(
        logger=logger,
        precision=config["precision"],
        max_epochs=config["max_epochs"],
        log_every_n_steps=config["log_every_n_steps"],
        default_root_dir=config["wandb_params"]["save_dir"],
        enable_model_summary=False,
        # callbacks=[checkpoint_callback]
        # deterministic=True
    )
    dl = get_dataloaders(config)
    trainer.fit(
        model=model, 
        train_dataloaders=dl["train_dataloaders"], 
        val_dataloaders=dl["val_dataloaders"][0]
    )
    # trainer.test(model, dataloaders=test_dataloader)

if __name__ == '__main__':
    config = load_config('dpt_run_config.yaml')
    seed_everything(config["seed"], workers=True)
    train(config)
