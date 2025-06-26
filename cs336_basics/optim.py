import math
from typing import Iterable, BinaryIO, IO
import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    """Minimal AdamW optimizer implemented from scratch."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        if lr < 0:
            raise ValueError("Invalid learning rate")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                step = state['step']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                denom = exp_avg_sq.sqrt().add_(eps)
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                if wd != 0:
                    p.data.add_(p.data, alpha=-lr * wd)
        return loss


def get_adamw_cls():
    return AdamW


def get_lr_cosine_schedule(it: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int) -> float:
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it > cosine_cycle_iters:
        return min_learning_rate
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_learning_rate + (max_learning_rate - min_learning_rate) * cosine


def save_checkpoint(model: torch.nn.Module, optimizer: Optimizer, iteration: int, out: str | BinaryIO | IO[bytes]):
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'iteration': iteration}, out)


def load_checkpoint(src: str | BinaryIO | IO[bytes], model: torch.nn.Module, optimizer: Optimizer) -> int:
    data = torch.load(src, map_location='cpu')
    model.load_state_dict(data['model'])
    optimizer.load_state_dict(data['optimizer'])
    return int(data['iteration'])
