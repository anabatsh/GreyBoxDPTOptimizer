from email.policy import default
import os
import yaml
import random
from functools import partial
from collections import defaultdict
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import collate, default_collate_fn_map

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import seed_everything, LightningDataModule
from lightning.pytorch.profilers import PyTorchProfiler

from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp import MixedPrecision

from scripts.create_problem import load_problem_set
from dpt.data import OfflineDataset, OnlineDataset
from dpt.train import DPTSolver
import problems as pbs

from tqdm.auto import tqdm
from dpt.model import DPT
from dpt.nn import TransformerBlock
import copy
import argparse

torch.backends.cuda.enable_flash_sdp(True)
torch.set_float32_matmul_precision("high")

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
    custom_collate_fn_map = default_collate_fn_map.copy()
    custom_collate_fn_map[problem_class] = collate_problem_fn
    return collate(batch, collate_fn_map=custom_collate_fn_map)

class ProblemDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def prepare_data(self):
        data_path = self.config['data_path']
        if not os.path.exists(data_path) or len(os.listdir(data_path)) == 0:
            raise ValueError(f'No data found in {data_path}. Run generate_data.py to create data.')

    def setup(self, stage=None):
        data_path = self.config['data_path']
        problem_names = os.listdir(data_path)
        for name in config['exclude_problems']:
            problem_names.remove(name)

        self.collate_fn = partial(custom_collate_fn, problem_class=getattr(pbs, 'Problem'))
        self.data = defaultdict(list)
        for suffix in ('train', 'val', 'test'):
            for problem in problem_names:
                read_path = os.path.join(data_path, problem, suffix)
                problems = load_problem_set(read_path)
                self.data[suffix].extend(problems)
            print(f'Train set: {len(self.data[suffix])} problems')

    def train_dataloader(self):
        train_offline_dataset = OfflineDataset(
            problems=self.data['train'],
            seq_len=self.config["model_params"]["seq_len"],
            action=self.config['action'],
            target_action=self.config['target_action'],
            ad_ratio=self.config['ad_ratio'],
            results_dir=self.config['trajectories_path'],
            suffix='train'
        )
        return DataLoader(
            dataset=train_offline_dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            pin_memory=True,
            shuffle=True,
            collate_fn=self.collate_fn,
            # prefetch_factor=8,
            # persistent_workers=True,
        )

    def val_dataloader(self):
        val_offline_dataset = OfflineDataset(
            problems=self.data['val'],
            seq_len=self.config["model_params"]["seq_len"],
            action=self.config['action'],
            target_action=self.config['target_action'],
            ad_ratio=self.config['ad_ratio'],
            results_dir=self.config['trajectories_path'],
            suffix='val'
        )
        return DataLoader(
                dataset=val_offline_dataset,
                batch_size=self.config["batch_size"],
                num_workers=self.config["num_workers"],
                pin_memory=True,
                shuffle=False,
                collate_fn=self.collate_fn,
            )

    def test_dataloader(self):
        val_online_dataset = OnlineDataset(
            problems=self.data['test']
        )
        return DataLoader(
            dataset=val_online_dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

def train(config):
    logger = WandbLogger(**config["wandb_params"])
    model = DPTSolver(config)
    datamodule = ProblemDataModule(config)
    auto_wrap_policy = {TransformerBlock}
    checkpoint_callback = ModelCheckpoint(every_n_epochs=25, save_last=True, filename='{epoch}')
    if config["strategy"] == "fsdp":
        strategy = FSDPStrategy(
            cpu_offload=False,
            sharding_strategy="SHARD_GRAD_OP",
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16, reduce_dtype=torch.float32),
        )
    else:
        strategy = config["strategy"]

    trainer = L.Trainer(
        logger=logger,
        precision=config["precision"] if config["strategy"] != "fsdp" else None,
        max_epochs=config["max_epochs"],
        log_every_n_steps=config["log_every_n_steps"],
        default_root_dir=config["wandb_params"]["save_dir"],
        enable_model_summary=True,
        callbacks=[checkpoint_callback],
        # use_distributed_sampler=False,
        strategy=strategy,
        # profiler=PyTorchProfiler(filename="profiler_output.txt"),
        # deterministic=True
    )
    trainer.fit(
        model=model,
        datamodule=datamodule,
        # ckpt_path="results/DPT_3/0ogshb6h/checkpoints/epoch=49.ckpt",
    )
    # trainer.test(model, datamodule=datamodule)

if __name__ == '__main__':
    # Set up argparse
    parser = argparse.ArgumentParser(description='Load configuration file.')
    parser.add_argument('config', type=str, nargs='?', default='configs/config_10.yaml', 
                        help='Path to the configuration file (default: config.yaml)')

    # Parse arguments
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    seed_everything(config["seed"], workers=True)
    train(config)