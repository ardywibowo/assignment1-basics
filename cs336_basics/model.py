import math
from dataclasses import dataclass
from typing import Optional
import torch
from torch import Tensor
from .nn_utils import softmax, gradient_clipping, cross_entropy


def linear(weights: Tensor, in_features: Tensor) -> Tensor:
    return in_features @ weights.t()


def embedding(weights: Tensor, token_ids: Tensor) -> Tensor:
    """Simple embedding lookup implemented from scratch."""
    return weights[token_ids]


def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


def swiglu(w1_weight: Tensor, w2_weight: Tensor, w3_weight: Tensor, in_features: Tensor) -> Tensor:
    x = in_features @ w1_weight.t()
    x = silu(x)
    v = in_features @ w3_weight.t()
    return (x * v) @ w2_weight.t()


def scaled_dot_product_attention(Q: Tensor, K: Tensor, V: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    d_k = Q.shape[-1]
    attn = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        attn = attn.masked_fill(~mask, float('-inf'))
    attn = softmax(attn, dim=-1)
    return attn @ V


def apply_rope(x: Tensor, token_positions: Tensor, theta: float, max_seq_len: int) -> Tensor:
    d = x.shape[-1]
    assert d % 2 == 0
    device = x.device
    pos = token_positions
    inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, device=device).float() / d))
    freqs = torch.einsum('...,j->...j', pos.float(), inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    out = torch.stack((out1, out2), dim=-1)
    return out.flatten(-2)


def multihead_self_attention(
    in_features: Tensor,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    num_heads: int,
    mask: Optional[Tensor] = None,
    theta: Optional[float] = None,
    token_positions: Optional[Tensor] = None,
    max_seq_len: Optional[int] = None,
) -> Tensor:
    batch, seq_len, d_model = in_features.shape
    d_head = d_model // num_heads
    Q = (in_features @ q_proj_weight.t()).view(batch, seq_len, num_heads, d_head).transpose(1, 2)
    K = (in_features @ k_proj_weight.t()).view(batch, seq_len, num_heads, d_head).transpose(1, 2)
    V = (in_features @ v_proj_weight.t()).view(batch, seq_len, num_heads, d_head).transpose(1, 2)
    if theta is not None and token_positions is not None and max_seq_len is not None:
        Q = apply_rope(Q, token_positions, theta, max_seq_len)
        K = apply_rope(K, token_positions, theta, max_seq_len)

    if mask is None:
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=in_features.device).tril()

    attn_out = scaled_dot_product_attention(Q, K, V, mask)
    attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
    return attn_out @ o_proj_weight.t()


def rmsnorm(in_features: Tensor, weight: Tensor, eps: float) -> Tensor:
    variance = in_features.pow(2).mean(dim=-1, keepdim=True)
    normed = in_features * torch.rsqrt(variance + eps)
    return normed * weight


def transformer_block(
    in_features: Tensor,
    weights: dict[str, Tensor],
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
) -> Tensor:
    x = rmsnorm(in_features, weights['ln1.weight'], eps=1e-5)
    attn_out = multihead_self_attention(
        x,
        weights['attn.q_proj.weight'],
        weights['attn.k_proj.weight'],
        weights['attn.v_proj.weight'],
        weights['attn.output_proj.weight'],
        num_heads=num_heads,
        theta=theta,
        token_positions=torch.arange(x.size(1), device=x.device),
        max_seq_len=max_seq_len,
    )
    x = in_features + attn_out
    y = rmsnorm(x, weights['ln2.weight'], eps=1e-5)
    ff_out = swiglu(
        weights['ffn.w1.weight'],
        weights['ffn.w2.weight'],
        weights['ffn.w3.weight'],
        y,
    )
    return x + ff_out


def transformer_lm(
    in_indices: Tensor,
    weights: dict[str, Tensor],
    num_layers: int,
    num_heads: int,
    d_ff: int,
    context_length: int,
    rope_theta: float,
    vocab_size: int,
    d_model: int,
) -> Tensor:
    x = embedding(weights['token_embeddings.weight'], in_indices)
    for layer in range(num_layers):
        prefix = f'layers.{layer}.'
        layer_weights = {k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}
        x = transformer_block(
            x,
            layer_weights,
            num_heads=num_heads,
            d_ff=d_ff,
            max_seq_len=context_length,
            theta=rope_theta,
        )
    x = rmsnorm(x, weights['ln_final.weight'], eps=1e-5)
    logits = linear(weights['lm_head.weight'], x)
    return logits
