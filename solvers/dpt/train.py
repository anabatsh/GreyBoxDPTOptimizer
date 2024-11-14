import os
import yaml
import torch
from torch.nn import functional as F
import lightning as L

try:
    from model import DPT_K2D
    from model_bo import DPT_K2D
    from utils.schedule import cosine_annealing_with_warmup
except ImportError:
    from .model import DPT_K2D
    from .model_bo import DPT_K2D
    from .utils.schedule import cosine_annealing_with_warmup

device = "cuda" if torch.cuda.is_available() else "cpu"


# def set_seed(seed: int):
#     os.environ["PYTHONHASHSEED"] = str(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.manual_seed(seed)

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Config file not found: {config_path}')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

class DPTSolver(L.LightningModule):
    def __init__(self, config_path):
        super().__init__()
        self.config = load_config(config_path)['TrainConfig']
        self.model = DPT_K2D(**self.config["model_params"]).to(device)
        self.save_hyperparameters()

    def get_action(self, logits):
        temp = self.config["temperature"]
        if temp <= 0:
            temp = 1.0
        probs = F.softmax(logits / temp, dim=-1)
        if not self.training and self.config["do_samples"]:
            action = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            action = torch.argmax(probs, dim=-1)
        return action

    def get_loss(self, logits, target_action):
        """
        if training:
            logits:        torch.Tensor # [batch_size, n_steps, num_actions]
            target_action: torch.Tensor # [batch_size]
        if test:
            logits:        torch.Tensor # [batch_size, num_actions]
            target_action: torch.Tensor # [batch_size]
        """
        if logits.ndim == 3:
            input = logits.transpose(1, 2)
            target = target_action.repeat(input.shape[-1], 1).transpose(0, 1)
        else:
            input = logits
            target = target_action
        loss = F.cross_entropy(input, target, label_smoothing=self.config["label_smoothing"])
        return loss
    
    def get_accuracy(self, logits, target_action):
        with torch.no_grad():
            action = self.get_action(logits)
            target = target_action[:, None] if action.ndim == 2 else target_action
            accuracy = (action == target).to(torch.float).mean()
        return accuracy
    
    def _offline_step(self, batch, batch_idx):
        logits = self.model(
            query_state=batch["query_state"].to(device),
            context_states=batch["states"].to(device),
            context_actions=batch["actions"].to(device),
            context_next_states=batch["next_states"].to(device),
            context_rewards=batch["rewards"].to(device),
        )
        target_action = batch["target_action"].to(device)
        loss = self.get_loss(logits, target_action)
        accuracy = self.get_accuracy(logits, target_action)
        return {"loss": loss, "accuracy": accuracy}

    def _online_step(self, batch, batch_idx):
        loss = []
        accuracy = []
        for sample in batch:
            sample_loss, sample_accuracy = self.run(**sample).values()
            loss.append(sample_loss)
            accuracy.append(sample_accuracy)
        loss = torch.tensor(loss).mean()
        accuracy = torch.tensor(accuracy).mean()
        return {"loss": loss, "accuracy": accuracy}

    def training_step(self, batch, batch_idx):
        loss, accuracy = self._offline_step(batch, batch_idx).values()
        self.log("train loss", loss, on_step=True, on_epoch=False)
        self.log("train accuracy", accuracy, on_step=True, on_epoch=False)
        return {"loss": loss, "accuracy": accuracy}

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        if isinstance(batch, dict) and "states" in batch:
            loss, accuracy = self._offline_step(batch, batch_idx).values()
        else:
            loss, accuracy = self._online_step(batch, batch_idx).values()
        return {"loss": loss, "accuracy": accuracy}
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, accuracy = self.test_step(batch, batch_idx, dataloader_idx).values()
        self.log(f"val loss", loss, on_step=False, on_epoch=True)
        self.log(f"val accuracy", accuracy, on_step=False, on_epoch=True)
        return {"loss": loss, "accuracy": accuracy}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
            betas=self.config["betas"],
        )
        if self.config["with_scheduler"]:
            scheduler = cosine_annealing_with_warmup(
                optimizer=optimizer,
                warmup_epochs=self.config["warmup_epochs"],
                total_epochs=self.config["max_epochs"],
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer

    def run(
            self, query_state, 
            transition_function, reward_function, 
            target_action=None, n_steps=10,
            return_trajectory=False
        ):
        """
        query_state:    torch.Tensor # [num_states],
        transition function: (state, action) -> new state
            state:      torch.Tensor # [num_states]
            action:     torch.Tensor # [1]
            new_state:  torch.Tensor # [num_states]
        reward_function: (states, actions, next_states) -> reward
            states:     torch.Tensor  # [n_steps, num_states]
            actions:    torch.Tensor  # [n_steps]
            new_states: torch.Tensor  # [n_steps, num_states]
            reward:     torch.Tensor  # [1]
        """
        num_actions = self.config["model_params"]["num_actions"]
        num_states = self.config["model_params"]["num_states"]
        assert query_state.shape[-1] == num_states

        # [1, num_states]
        query_state = query_state.to(dtype=torch.float, device=device).unsqueeze(0)
        # [1, 0, num_states]
        states = torch.Tensor(1, 0, num_states).to(dtype=torch.float, device=device)
        # [1, 0, num_states]
        next_states = torch.Tensor(1, 0, num_states).to(dtype=torch.float, device=device)
        # [1, 0, num_actions]
        logits = torch.Tensor(1, 0, num_actions).to(dtype=torch.long, device=device)
        # [1, 0]
        actions = torch.Tensor(1, 0).to(dtype=torch.long, device=device)
        # [1, 0]
        rewards = torch.Tensor(1, 0).to(dtype=torch.float, device=device)
        
        for _ in range(n_steps):
            # [1, num_actions]
            predicted_logits = self.model(
                query_state=query_state,
                context_states=states,
                context_next_states=next_states,
                context_actions=actions,
                context_rewards=rewards,
            )
            # [1]
            predicted_action = self.get_action(predicted_logits)
            # [1, num_states]
            state = transition_function(
                query_state.cpu(), predicted_action.cpu()
            ).to(dtype=torch.float, device=device)
            # [1, n_step, num_states]
            states = torch.cat([states, query_state.unsqueeze(1)], dim=1)
            # [1, n_step, num_states]
            next_states = torch.cat([next_states, state.unsqueeze(1)], dim=1)
            # [1, n_step, num_actions]
            logits = torch.cat([logits, predicted_logits.unsqueeze(1)], dim=1)
            # [1, n_step]
            actions = torch.cat([actions, predicted_action.unsqueeze(1)], dim=1)
            # [1]
            reward = reward_function(
                states.cpu(), actions.cpu(), next_states.cpu(),
            ).to(dtype=torch.float, device=device)
            # [1, n_step]
            rewards = torch.cat([rewards, reward.unsqueeze(1)], dim=1)
            # [1, n_step]
            query_state = state

        result = {}

        if return_trajectory:
            result |= {
                "query_state": query_state.cpu()[0],
                "states": states.cpu()[0],
                "logits": logits.cpu()[0],
                "actions": actions.cpu()[0],
                "next_states": next_states.cpu()[0],
                "rewards": rewards.cpu()[0],
            }

        if target_action is not None:
            target_action = target_action.to(device).unsqueeze(0)
            loss = self.get_loss(logits, target_action)
            accuracy = (actions == target_action[:, None]).to(torch.float).mean()
            # accuracy = self.get_accuracy(logits, target_action)
            result |= {"loss": loss, "accuracy": accuracy}

        return result