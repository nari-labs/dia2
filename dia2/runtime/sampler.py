from __future__ import annotations

import torch

@torch.jit.script
def sample_token(
    logits: torch.Tensor,
    temp: float,
    top_k: int = 0,
) -> torch.Tensor:
    logits32 = logits.to(torch.float32)
    if temp <= 0.0:
        return torch.argmax(logits32, dim=-1, keepdim=True)
    probs = torch.softmax(logits32 / max(temp, 1e-6), dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    probs = torch.clamp_min(probs, 0.0)
    flat = probs.reshape(-1, probs.shape[-1])
    norm = flat.sum(dim=-1, keepdim=True)
    norm = norm.clamp_min(1e-12)
    flat = flat / norm
    zero_mask = norm <= 0
    if zero_mask.any():
        filler = torch.zeros_like(flat)
        filler[..., 0] = 1.0
        mask = zero_mask.expand_as(flat)
        flat = torch.where(mask, filler, flat)
    vocab = flat.shape[-1]
    if top_k > 0 and top_k < vocab:
        topv, indices = torch.topk(flat, top_k, dim=-1)
        topv = topv / topv.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        draws = torch.multinomial(topv, num_samples=1)
        picks = torch.gather(indices, dim=-1, index=draws)
    else:
        picks = torch.multinomial(flat, num_samples=1)
    picks = picks.reshape([i for i in probs.shape[:-1]] + [1])
    return picks
