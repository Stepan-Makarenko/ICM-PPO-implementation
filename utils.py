import torch.nn as nn


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * nn.Sigmoid()(x)


def linear_decay_lr(optimizer, n_step, max_step=1e7, max_lr=3e-4, min_lr=1e-5):
    if n_step >= max_step:
        optimizer.param_groups[0]['lr'] = min_lr
    else:
        optimizer.param_groups[0]['lr'] = (min_lr - max_lr) / max_step * n_step + max_lr


def linear_decay_beta(n_step, max_step=1e7, max_b=1e-2, min_b=1e-5):
    if n_step >= max_step:
        return min_b
    else:
        return (min_b - max_b) / max_step * n_step + max_b


def linear_decay_eps(n_step, max_step=1e7, max_e=0.2, min_e=0.1):
    if n_step >= max_step:
        return min_e
    else:
        return (min_e - max_e) / max_step * n_step + max_e

