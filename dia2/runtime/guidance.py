from __future__ import annotations

import torch

from .sampler import sample_token


def apply_classifier_guidance(
    logits: torch.Tensor,
    cfg_active: bool,
    scale: float,
    top_k: int,
) -> torch.Tensor:
    if not cfg_active:
        return logits
    conditional = logits[0:1]
    unconditional = logits[1:2]
    cond32 = conditional.to(torch.float32)
    uncond32 = unconditional.to(torch.float32)
    guided = torch.lerp(uncond32, cond32, scale)
    if top_k > 0 and guided.shape[-1] > 0:
        k = min(top_k, guided.shape[-1])
        threshold = torch.topk(guided, k=k, dim=-1, sorted=False).values[..., -1:]
        mask = guided >= threshold
        neg_inf = torch.full_like(cond32, float("-inf"))
        cond32 = torch.where(mask, cond32, neg_inf)
    return cond32.to(conditional.dtype)


def sample_audio_logits(logits: torch.Tensor, temp: float, top_k: int) -> torch.Tensor:
    """Sample a single audio token (shape [1]) from logits."""
    return (
        sample_token(
            logits,
            temp=temp,
            top_k=top_k,
        ).view(1)
    )
