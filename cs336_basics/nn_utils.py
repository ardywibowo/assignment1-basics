import math
from typing import Iterable
import torch
from torch import Tensor


def softmax(in_features: Tensor, dim: int) -> Tensor:
    """Numerically stable softmax implementation."""
    shifted = in_features - in_features.amax(dim=dim, keepdim=True)
    exp = torch.exp(shifted)
    return exp / exp.sum(dim=dim, keepdim=True)


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """Compute cross entropy loss from scratch."""
    shifted = logits - logits.amax(dim=-1, keepdim=True)
    log_exp_sum = torch.log(torch.exp(shifted).sum(dim=-1))
    gather = shifted.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    loss = log_exp_sum - gather
    return loss.mean()


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_norm: float) -> None:
    """Clip gradients in-place to have global norm at most ``max_norm``."""
    params = [p for p in parameters if p.requires_grad and p.grad is not None]
    if not params:
        return
    device = params[0].grad.device
    total_norm = torch.sqrt(
        sum((p.grad.detach() ** 2).sum() for p in params)
    )
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in params:
            p.grad.detach().mul_(clip_coef)
