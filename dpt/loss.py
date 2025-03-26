import torch
from torch import nn
from torch.nn import functional as F


class BCELoss(nn.Module):
    def __init__(self, label_smoothing=0.0): 
        super().__init__()
        self.smoothing = label_smoothing
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets, *args, **kwargs):
        """
        outputs - [batch_size, seq_len + 1, action_dim]
        targets - [batch_size, seq_len + 1, action_dim]
        """
        # Reshape to [batch_size * (seq_len + 1), action_dim]
        logits = outputs.view(-1, outputs.size(-1)).contiguous()
        labels = targets.view(-1, outputs.size(-1)).float().contiguous()

        # Apply label smoothing
        if self.smoothing > 0:
            labels = labels * (1 - self.smoothing) + 0.5 * self.smoothing  # Smooth labels towards 0.5

        return {"loss": self.loss_fn(logits, labels)}


class CELoss(nn.Module):
    def __init__(self, label_smoothing=0.0): 
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, outputs, targets, *args, **kwargs):
        """
        outputs - [batch_size, seq_len + 1, action_dim]
        targets - [batch_size, seq_len + 1, action_dim]
        """        
        # Reshape to [batch_size, action_dim, seq_len + 1]
        logits = outputs.permute(0, 2, 1)
        labels = targets.permute(0, 2, 1).float()
        return {"loss": self.loss_fn(logits, labels)}


class RKLLoss(nn.Module):
    def __init__(self, label_smoothing=0.0): 
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, outputs, targets, *args, **kwargs):
        """
        outputs - [batch_size, seq_len + 1, action_dim]
        targets - [batch_size, seq_len + 1, action_dim]
        """
        # [batch_size, seq_len + 1, action_dim]
        q = F.softmax(outputs, dim=-1)

        # [batch_size, seq_len + 1, action_dim]
        labels = targets.float()
        p = labels / labels.sum(dim=-1, keepdim=True)

        loss_qq = (q * torch.log(q + 1e-6)).sum(-1)
        loss_qp = (q * torch.log(p + 1e-6)).sum(-1)

        loss = (loss_qq - loss_qp).mean()
        return {"loss": loss}