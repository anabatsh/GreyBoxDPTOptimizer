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
from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch import seed_everything, LightningDataModule

from src.data import OfflineDataset, OnlineDataset
from src.train import DPTSolver
import problems as pbs

from tqdm.auto import tqdm
from src.nn import TransformerBlock

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
        problem_class = getattr(pbs, self.config["problem_params"]["problem"])
        data_path = "__".join(f"{k}_{v}" for k, v in sorted(self.config["problem_params"].items())) + ".dill"
        data_path = os.path.join("data", data_path)
        if not os.path.exists(data_path):
            print("Generating dataset...")
            problem_set = pbs.ProblemSet([problem_class(**self.config["problem_params"], seed=i) for i in tqdm(range(self.config["problem_params"]["n_problems"]))])
            pbs.serialize_problem_set(problem_set, data_path)
            print(f"Saved problem set to {data_path}")

    def setup(self, stage=None):
        data_path = "__".join(f"{k}_{v}" for k, v in sorted(self.config["problem_params"].items())) + ".dill"
        data_path = os.path.join("data", data_path)
        self.problems = pbs.deserialize_problem_set(data_path).problems[:self.config["problem_params"]]["use_problems"]
        # This method is called on every GPU, but the data is already prepared
        val_size = int(len(self.problems) * 0.1)
        random.shuffle(self.problems)
        self.train_problems = self.problems[:-val_size]
        self.val_problems = self.problems[-val_size:]

    def train_dataloader(self):
        collate_fn = partial(custom_collate_fn, problem_class=getattr(pbs, self.config["problem_params"]["problem"]))
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
            collate_fn=collate_fn
        )

    def val_dataloader(self):
        collate_fn = partial(custom_collate_fn, problem_class=getattr(pbs, self.config["problem_params"]["problem"]))
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
                collate_fn=collate_fn
            )

    def test_dataloader(self):
        collate_fn = partial(custom_collate_fn, problem_class=getattr(pbs, self.config["problem_params"]["problem"]))
        val_online_dataset = OnlineDataset(self.val_problems)
        return DataLoader(
            dataset=val_online_dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            pin_memory=True,
            shuffle=False,
            collate_fn=collate_fn
        )

def train(config):
    logger = WandbLogger(**config["wandb_params"])
    model = DPTSolver(config)
    datamodule = ProblemDataModule(config)
    fsdp_wrap_policy = {TransformerBlock}
    checkpoint_callback = ModelCheckpoint(every_n_epochs=50, save_top_k=-1, filename='{epoch}')
    trainer = L.Trainer(
        logger=logger,
        precision=config["precision"],
        max_epochs=config["max_epochs"],
        log_every_n_steps=config["log_every_n_steps"],
        default_root_dir=config["wandb_params"]["save_dir"],
        enable_model_summary=True,
        callbacks=[checkpoint_callback],
        use_distributed_sampler=False,
        strategy=FSDPStrategy(sharding_strategy="SHARD_GRAD_OP", auto_wrap_policy=fsdp_wrap_policy),
        # strategy="ddp",
        # deterministic=True
    )
    trainer.fit(
        model=model,
        datamodule=datamodule,
    )
    trainer.test(model, datamodule=datamodule)

if __name__ == '__main__':
    config = load_config('config.yaml')
    seed_everything(config["seed"], workers=True)
    train(config)
