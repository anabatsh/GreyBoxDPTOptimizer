from email.policy import default
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

from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp import MixedPrecision

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
        self.train_problems = None
        self.test_problems = None
        self.val_problems = None
        self.test_size = 1000

    def prepare_data(self):
        # This method is called only once, regardless of the number of GPUs
        problem_class = getattr(pbs, self.config['problem_params']["problem"])
        params = "__".join(f"{k}_{v}" for k, v in self.config["problem_params"].items() if k != "use_problems")
        data_path = os.path.join(self.config["data_path"], params + ".dill")
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)
            print("Generating dataset...")
            problem_set = pbs.ProblemSet([problem_class(**self.config["problem_params"], seed=i) for i in tqdm(range(self.config["problem_params"]["n_problems"]))])
            pbs.serialize_problem_set(problem_set, data_path)
            print(f"Saved problem set to {data_path}")
            test_problem_set = copy.copy(problem_set)
            test_problem_set.problems = test_problem_set.problems[-self.test_size:]
            pbs.serialize_problem_set(test_problem_set, data_path.split(".dill")[0] + "__test.dill")
            print(f"Saved test set to {data_path}")

    def setup(self, stage=None):
        data_path = "__".join(f"{k}_{v}" for k, v in sorted(self.config["problem_params"].items()) if k != "use_problems") + ".dill"
        data_path = os.path.join("data", data_path)
        self.problems = pbs.deserialize_problem_set(data_path).problems[-self.config["problem_params"]["use_problems"]:]
        problem_class = getattr(pbs, self.config["problem_params"]["problem"])
        # for i in range(len(self.problems)):
        #     new_problem = problem_class.__new__(problem_class)
        #     new_problem.__dict__ =self.problems[i].__dict__
        #     self.problems[i] = new_problem
        self.collate_fn = partial(custom_collate_fn, problem_class=problem_class)
        # This method is called on every GPU, but the data is already prepared
        val_size = 5000
        test_size = 1000
        # random.shuffle(self.problems)
        self.train_problems = self.problems[:-val_size-test_size]
        self.val_problems = self.problems[-val_size-test_size:-test_size]
        self.test_problems = self.problems[-test_size:]

    def train_dataloader(self):
        train_offline_dataset = OfflineDataset(
            problems=self.train_problems,
            seq_len=self.config["model_params"]["seq_len"],
            ad_eps=self.config["ad_eps"],
        )
        return DataLoader(
            dataset=train_offline_dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            pin_memory=True,
            shuffle=True,
            collate_fn=self.collate_fn,
            prefetch_factor=8,
            persistent_workers=True,
        )

    def val_dataloader(self):
        val_offline_dataset = OfflineDataset(
            problems=self.val_problems,
            seq_len=self.config["model_params"]["seq_len"],
            ad_eps=[0.5, 0.5],
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
        val_online_dataset = OnlineDataset(self.test_problems)
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
    checkpoint_callback = ModelCheckpoint(every_n_epochs=25, save_last=10, save_top_k=-1, filename='{epoch}')
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
        use_distributed_sampler=False,
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
    parser.add_argument('config', type=str, nargs='?', default='config.yaml', 
                        help='Path to the configuration file (default: config.yaml)')

    # Parse arguments
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    seed_everything(config["seed"], workers=True)
    train(config)