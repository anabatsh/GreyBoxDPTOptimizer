import os
import yaml
import random
from functools import partial
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import collate, default_collate_fn_map

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import seed_everything, LightningDataModule

from dpt import *
import problems as pbs

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

class ProblemDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_problems = None
        self.val_problems = None

    def prepare_data(self):
        # This method is called only once, regardless of the number of GPUs
        problem_class = getattr(pbs, self.config["problem"])
        params = "_".join(f"{k}_{v}" for k, v in self.config["problem_params"].items())
        data_path = os.path.join(self.config["data_path"], self.config["problem"] + '_' + params)
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)
            print("Generating dataset...")
            n_train = self.config["n_train_problems"]
            n_val = self.config["n_val_problems"]
            n_test = self.config["n_test_problems"]
            problems = [problem_class(**self.config["problem_params"], seed=i) for i in tqdm(range(n_train + n_val + n_test))]
            train_problemset = pbs.ProblemSet(problems[:n_train])
            val_problemset = pbs.ProblemSet(problems[n_train:n_train + n_val])
            test_problemset = pbs.ProblemSet(problems[n_train + n_val:])
            pbs.serialize_problem_set(train_problemset, os.path.join(data_path, "train.dill"))
            pbs.serialize_problem_set(val_problemset, os.path.join(data_path, "val.dill"))
            pbs.serialize_problem_set(test_problemset, os.path.join(data_path, "test.dill"))
            print(f"Saved problem sets to {data_path}")

    def setup(self, stage=None):
        # This method is called on every GPU, but the data is already prepared
        problem_class = getattr(pbs, self.config["problem"])
        self.collate_fn = partial(custom_collate_fn, problem_class=problem_class)

        params = "_".join(f"{k}_{v}" for k, v in self.config["problem_params"].items())
        data_path = os.path.join(self.config["data_path"], self.config["problem"] + '_' + params)

        self.train_problems = pbs.deserialize_problem_set(os.path.join(data_path, "train.dill")).problems
        self.val_problems = pbs.deserialize_problem_set(os.path.join(data_path, "val.dill")).problems
        self.test_problems = pbs.deserialize_problem_set(os.path.join(data_path, "test.dill")).problems
    
        print(f'train_problems: {len(self.train_problems)}')
        print(f'val_problems: {len(self.val_problems)}')
        print(f'test_problems: {len(self.test_problems)}')

    def train_dataloader(self):
        train_offline_dataset = OfflineDataset(
            problems=self.train_problems,
            seq_len=self.config["model_params"]["seq_len"]
        )
        return DataLoader(
            dataset=train_offline_dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            pin_memory=True,
            shuffle=True,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        val_offline_dataset = OfflineDataset(
            problems=self.val_problems,
            seq_len=self.config["model_params"]["seq_len"]
        )
        return DataLoader(
            dataset=val_offline_dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        val_online_dataset = OnlineDataset(self.val_problems)
        return DataLoader(
            dataset=val_online_dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn
        )

def train(config):
    logger = WandbLogger(**config["wandb_params"])
    model = DPTSolver(config)
    datamodule = ProblemDataModule(config)

    # checkpoint_callback = ModelCheckpoint(every_n_epochs=50, save_top_k=-1, filename='{epoch}')
    trainer = L.Trainer(
        logger=logger,
        precision=config["precision"],
        max_epochs=config["max_epochs"],
        log_every_n_steps=config["log_every_n_steps"],
        default_root_dir=config["wandb_params"]["save_dir"],
        enable_model_summary=True,
        # callbacks=[checkpoint_callback]
        # deterministic=True
    )
    trainer.fit(model=model, datamodule=datamodule)
    # trainer.test(model, datamodule=datamodule)

if __name__ == '__main__':
    config = load_config('config.yaml')
    seed_everything(config["seed"], workers=True)
    train(config)