import torch
import math
import functools


# source:  https://gist.github.com/akshaychawla/86d938bc6346cf535dce766c83f743ce
def _cosine_decay_warmup(iteration, warmup_iterations, total_iterations):
    """
    Linear warmup from 0 --> 1.0, then decay using cosine decay to 0.0
    """
    if iteration <= warmup_iterations:
        multiplier = iteration / warmup_iterations
    else:
        multiplier = (iteration - warmup_iterations) / (total_iterations - warmup_iterations)
        multiplier = 0.5 * (1 + math.cos(math.pi * multiplier))
    return multiplier


def _constant_warmup(iteration, warmup_iterations):
    """
    Linear warmup from 0 --> 1.0, then constant
    """
    multiplier = 1.0
    if iteration <= warmup_iterations:
        multiplier = iteration / warmup_iterations
    return multiplier


def cosine_annealing_with_warmup(optimizer, warmup_steps, total_steps):
    _decay_func = functools.partial(
        _cosine_decay_warmup,
        warmup_iterations=warmup_steps,
        total_iterations=total_steps
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _decay_func)
    return scheduler


def linear_warmup(optimizer, warmup_steps):
    _decay_func = functools.partial(
        _constant_warmup,
        warmup_iterations=warmup_steps
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _decay_func)
    return scheduler

def cosine_annealing_with_warmup(optimizer, warmup_epochs, total_epochs):
    from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
    warmup_lr_lambda = lambda epoch: (epoch + 1) / warmup_epochs if epoch < warmup_epochs else 1
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lr_lambda)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(total_epochs - warmup_epochs), eta_min=0)
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
    )
    return scheduler