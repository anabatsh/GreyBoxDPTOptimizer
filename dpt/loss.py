from torch import nn


class BCELoss(nn.Module):
    def __init__(self, label_smoothing=0.0): 
        super().__init__()
        self.smoothing = label_smoothing
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets, *args, **kwargs):
        """
        outputs - [batch_size, seq_len + 1, action_dim]
        targets - [batch_size, action_dim]
        """
        # Binary Cross-Entropy for parallel tasks

        # Reshape outputs to [batch_size * (seq_len + 1), action_dim]
        logits = outputs.view(-1, outputs.size(-1)).contiguous()

        # Repeat targets to match the sequence length dimension
        # targets will be [batch_size, seq_len + 1, action_dim] after repeat
        targets = targets.float().unsqueeze(1).repeat(1, outputs.size(1), 1)

        # Reshape targets to [batch_size * (seq_len + 1), action_dim]
        labels = targets.view(-1, targets.size(-1)).contiguous()

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
        targets - [batch_size, action_dim]
        """
        # Cross-Entropy for parallel tasks

        # Reshape outputs to [batch_size, action_dim, seq_len + 1]
        logits = outputs.permute(0, 2, 1)

        # Repeat targets to match the outputs shape [batch_size, action_dim, seq_len + 1]
        labels = targets.unsqueeze(-1).repeat(1, 1, outputs.size(1)).float()

        return {"loss": self.loss_fn(logits, labels)}