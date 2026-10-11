from functools import partial
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .attention import AttentionBlock


def iterative_pinv(softmax_mat: torch.Tensor, n_iter: int = 6) -> torch.Tensor:
    """
    Computing the Moore-Penrose inverse via Schulz iteration.
    (Razavi et al. 2014)
    """
    i = torch.eye(
        softmax_mat.size(-1), device=softmax_mat.device, dtype=softmax_mat.dtype
    )
    k = softmax_mat

    # Exact coefficient computation: 1 / ||K||_1
    sum_k = torch.sum(k, dim=-2)
    max_sum_k = torch.max(sum_k, dim=-1, keepdim=True).values.unsqueeze(-1)
    v = (1.0 / (max_sum_k + 1e-7)) * k.transpose(-1, -2)

    for _ in range(n_iter):
        kv = torch.matmul(k, v)
        v = torch.matmul(
            0.25 * v,
            13 * i - torch.matmul(kv, 15 * i - torch.matmul(kv, 7 * i - kv)),
        )
    return v


class AvgPool(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (..., seq_len, head_dim)
        seq_len = x.shape[-2]
        head_dim = x.shape[-1]
        segments = seq_len // self.n
        if segments == 0:
            return x

        if seq_len % self.n == 0:
            return x.reshape(*x.shape[:-2], self.n, segments, head_dim).mean(dim=-2)

        n_round = self.n - seq_len % self.n
        x_avg_round = (
            x[..., : n_round * segments, :]
            .reshape(*x.shape[:-2], n_round, segments, head_dim)
            .mean(dim=-2)
        )
        x_avg_off = (
            x[..., n_round * segments :, :]
            .reshape(*x.shape[:-2], self.n - n_round, segments + 1, head_dim)
            .mean(dim=-2)
        )
        return torch.cat((x_avg_round, x_avg_off), dim=-2)


class NystromAttention(nn.Module):
    def __init__(
        self,
        dropout: float = 0.0,
        num_heads: int = 4,
        num_landmarks: int = 128,
        inv_iterations: int = 6,
        conv_kernel_size: Optional[int] = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.num_heads = num_heads
        self.inv_iterations = inv_iterations
        self.attn_drop = nn.Dropout(dropout)
        self.landmark_pooling = AvgPool(n=self.num_landmarks)

        self.skip_connection = None
        if conv_kernel_size is not None:
            self.skip_connection = nn.Conv2d(
                in_channels=self.num_heads,
                out_channels=self.num_heads,
                kernel_size=(conv_kernel_size, 1),
                padding=(conv_kernel_size // 2, 0),
                bias=False,
                groups=self.num_heads,
            )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        # Expected input shape: (B, S, H, D)
        # Transpose to (B, H, S, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = 1.0 / math.sqrt(q.size(-1))
        seq_len = q.size(-2)

        if self.num_landmarks >= seq_len:
            attn = torch.matmul(q * scale, k.transpose(-2, -1))
            if key_padding_mask is not None:
                attn = attn + key_padding_mask
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            x = torch.matmul(attn, v)
        else:
            q_landmarks = self.landmark_pooling(q)
            k_landmarks = self.landmark_pooling(k)

            kernel_1 = F.softmax(torch.matmul(q * scale, k_landmarks.transpose(-2, -1)), dim=-1)
            kernel_2 = F.softmax(torch.matmul(q_landmarks * scale, k_landmarks.transpose(-2, -1)), dim=-1)
            kernel_3 = F.softmax(torch.matmul(q_landmarks * scale, k.transpose(-2, -1)), dim=-1)

            kernel_2_inv = iterative_pinv(kernel_2, self.inv_iterations)

            kernel_3_v = torch.matmul(kernel_3, v)
            x = torch.matmul(torch.matmul(kernel_1, kernel_2_inv), kernel_3_v)

        if self.skip_connection is not None:
            x = x + self.skip_connection(v)

        x = self.attn_drop(x)
        # Transpose back to (B, S, H, D)
        x = x.transpose(1, 2)
        return x


class NystromBlock(AttentionBlock):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        expansion: int = 4,
        dropout: float = 0.0,
        cosine: bool = False,
        gated: bool = False,
        layer_scale: float = 1.0,
        context_dim: int | None = None,
    ):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            expansion=expansion,
            dropout=dropout,
            cosine=cosine,
            gated=gated,
            layer_scale=layer_scale,
            context_dim=context_dim,
        )
        self.attention_fn = NystromAttention(
            num_landmarks=128, num_heads=num_heads, dropout=dropout
        )

    def attn(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        pos_embed: torch.Tensor | None = None,
        pos_embed_context: torch.Tensor | None = None,
        rope: nn.Module | None = None,
    ) -> torch.Tensor:
        x = self.norm_attnx(x)
        context = self.norm_attnctx(context)
        k, v = rearrange(
            self.kv(context), "b n (kv h d) -> b n h d kv", h=self.num_heads, kv=2
        ).unbind(dim=-1)
        q = rearrange(self.q(x), "b n (h d) -> b n h d", h=self.num_heads)

        if rope is not None:
            q = rope(q)
            k = rope(k)
        else:
            if pos_embed is not None:
                pos_embed = rearrange(
                    pos_embed, "b n (h d) -> b n h d", h=self.num_heads
                )
                q = q + pos_embed
            if pos_embed_context is not None:
                pos_embed_context = rearrange(
                    pos_embed_context, "b n (h d) -> b n h d", h=self.num_heads
                )
                k = k + pos_embed_context

        if self.cosine:
            q, k = map(partial(F.normalize, p=2, dim=-1), (q, k))  # cosine sim
        x = self.attention_fn(q, k, v, key_padding_mask=attn_bias)
        x = rearrange(x, "b n h d -> b n (h d)")
        x = self.out(x)
        return x
