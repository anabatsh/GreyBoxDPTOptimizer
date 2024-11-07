import os
from dataclasses import asdict, dataclass
from typing import Optional, Tuple
import random
import numpy as np
import pyrallis
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import wandb
from tqdm.autonotebook import tqdm

from src.model_dpt import DPT_K2D
from src.utils.data import MarkovianDataset
from src.utils.schedule import cosine_annealing_with_warmup

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainConfig:
    # wandb params
    project: str = "GreyBoxDPT"
    group: str = "markoviann"
    name: str = None
    entity: str = None

    # model params
    num_states: int = 1    # CUSTOM
    num_actions: int = 16  # CUSTOM
    hidden_dim: int = 512
    num_layers: int = 4
    num_heads: int = 4
    seq_len: int = 50
    attention_dropout: float = 0.5
    residual_dropout: float = 0.1
    embedding_dropout: float = 0.3
    normalize_qk: bool = False
    pre_norm: bool = True

    # training params
    learning_rate: float = 3e-4
    warmup_ratio: float = 0.05
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-4
    clip_grad: Optional[float] = None
    batch_size: int = 16 #128
    update_steps: int = 50#00 #125_000
    num_workers: int = 0
    label_smoothing: float = 0.0

    # evaluation params
    eval_every: int = 10_000
    eval_episodes: int = 200
    eval_train_goals: int = 20
    eval_test_goals: int = 50

    # general params
    learning_histories_path: str = "solvers/dpt/trajectories"
    checkpoints_path: Optional[str] = "solvers/dpt/checkpoints"
    train_seed: int = 42
    data_seed: int = 0
    eval_seed: int = 42

    rnn_hidden: int = 1
    rnn_dropout: float = 0.0
    subsample: int = 1

    with_scheduler: bool = False
    with_prior: bool = False
    with_key_indicator: bool = True

def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

@pyrallis.wrap()
def train(config: TrainConfig):
    set_seed(config.train_seed)

    dict_config = asdict(config)
    # dict_config["mlc_job"] = os.getenv("PLATFORM_JOB_NAME")
    # wandb.init(
    #     project=config.project,
    #     group=config.group,
    #     name=config.name,
    #     entity=config.entity,
    #     config=dict_config,
    # )

    if not os.path.exists(config.learning_histories_path):
        from src.utils.data import results2trajectories
        results2trajectories(read_dir='results', save_dir=config.learning_histories_path)

    dataset = MarkovianDataset(config.learning_histories_path, seq_len=config.seq_len)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=config.num_workers,
    )

    # model & optimizer & scheduler setup
    model = DPT_K2D(
        num_states=config.num_states,
        num_actions=config.num_actions,
        hidden_dim=config.hidden_dim,
        seq_len=config.seq_len,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        attention_dropout=config.attention_dropout,
        residual_dropout=config.residual_dropout,
        embedding_dropout=config.embedding_dropout,
        normalize_qk=config.normalize_qk,
        pre_norm=config.pre_norm,
        rnn_weights_path=os.path.join(config.learning_histories_path, "rnn.pth"),
        state_rnn_embedding=config.rnn_hidden,
        rnn_dropout=config.rnn_dropout,
    ).to(DEVICE)

    # if needed, test beforehand
    # model = torch.compile(model)

    optim = torch.optim.AdamW(
        params=model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
    )

    current_lr = config.learning_rate
    if config.with_scheduler:
        scheduler = cosine_annealing_with_warmup(
            optimizer=optim,
            warmup_steps=int(config.update_steps * config.warmup_ratio),
            total_steps=config.update_steps,
        )

    # save config to the checkpoint
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    scaler = torch.amp.GradScaler('cuda')
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    for global_step, batch in tqdm(enumerate(dataloader), 'Training'):
        if global_step > config.update_steps:
            break

        (
            query_states,
            states,
            actions,
            next_states,
            rewards,
            target_actions,
        ) = [b.to(DEVICE) for b in batch]

        query_states = query_states.to(torch.float)
        states = states.to(torch.float)
        actions = actions.to(torch.long)
        next_states = next_states.to(torch.float)
        rewards = rewards.to(torch.float32)

        target_actions = target_actions.squeeze(-1)
        if config.with_prior:
            target_actions = (
                F.one_hot(target_actions, num_classes=config.num_actions)
                .unsqueeze(1)
                .repeat(1, config.seq_len + 1, 1)
                .float()
            )
        else:
            target_actions = (
                F.one_hot(target_actions, num_classes=config.num_actions)
                .unsqueeze(1)
                .repeat(1, config.seq_len, 1)
                .float()
            )

        with torch.amp.autocast('cuda'):
            predicted_actions = model(
                query_states=query_states,
                context_states=states,
                context_next_states=next_states,
                context_actions=actions,
                context_rewards=rewards,
            )

            if not config.with_prior:
                predicted_actions = predicted_actions[:, 1:, :]

            loss = F.cross_entropy(
                input=predicted_actions.flatten(0, 1),
                target=target_actions.flatten(0, 1),
                label_smoothing=config.label_smoothing,
            )
            # print(f'step: {global_step} loss:{loss.item()}')

        scaler.scale(loss).backward()
        if config.clip_grad is not None:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        scaler.step(optim)
        scaler.update()
        optim.zero_grad(set_to_none=True)
        if config.with_scheduler:
            scheduler.step()

        with torch.no_grad():
            a = torch.argmax(predicted_actions.flatten(0, 1), dim=-1)
            t = torch.argmax(target_actions.flatten(0, 1), dim=-1)
            accuracy = torch.sum(a == t) / a.shape[0]

            if config.with_scheduler:
                current_lr = scheduler.get_last_lr()[0]

            # wandb.log(
            #     {
            #         "loss": loss.item(),
            #         "accuracy": accuracy,
            #         "lr": current_lr
            #     },
            #     step=global_step,
            # )

            if global_step % config.eval_every == 0:
                model.eval()
                if config.checkpoints_path is not None:
                    torch.save(
                        model.state_dict(),
                        os.path.join(
                            config.checkpoints_path, f"model_{global_step}.pt"
                        ),
                    )
                model.train()

    if config.checkpoints_path is not None:
        torch.save(
            model.state_dict(), os.path.join(config.checkpoints_path, f"model_last.pt")
        )


if __name__ == "__main__":
    train()
