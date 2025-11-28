from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from dia2.config import DiaConfig


class RotaryEmbedding(nn.Module):
    def __init__(
      self,
      head_dim: int,
      min_timescale: int,
      max_timescale: int,
      max_seq_len: int = 1500,
      device: Optional[torch.device] = None,
    ):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE dimension must be even")
        half_dim = head_dim // 2
        fraction = (2.0 * torch.arange(0, half_dim)) / head_dim
        timescale = min_timescale * (max_timescale / min_timescale) ** fraction
        inv_freq = 1.0 / timescale
        self.max_seq_len = max_seq_len
        self.register_buffer("inv_freq", inv_freq.to(dtype=torch.float32, device=device), persistent=False)
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)
        self._update_cos_sin_cache()

    def _update_cos_sin_cache(self) -> None:
        t = torch.arange(self.max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        cos = self.cos_cached[position_ids].unsqueeze(2).to(dtype=x.dtype)
        sin = self.sin_cached[position_ids].unsqueeze(2).to(dtype=x.dtype)
        x1, x2 = torch.chunk(x, 2, dim=-1)
        rotated = torch.cat((-x2, x1), dim=-1)
        return (x * cos) + (rotated * sin)

def _get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name in ("silu", "swish", "swiglu"):
        return nn.SiLU()
    if name in ("gelu", "geglu"):
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "linear":
        return nn.Identity()
    raise ValueError(f"Unsupported activation {name}")


@dataclass
class AttentionShape:
    dim: int
    heads: int
    kv_heads: int
    head_dim: int
    rope_min: int
    rope_max: int
    apply_rope: bool


class Attention(nn.Module):
    """Byte-for-byte port of dia_v2 Attention.forward_incremental."""

    def __init__(
        self,
        config: DiaConfig,
        dim: int,
        compute_dtype: torch.dtype,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        dec = config.model.decoder
        self.num_query_heads = dec.gqa_query_heads
        self.num_kv_heads = dec.kv_heads
        self.head_dim = dec.gqa_head_dim
        self.num_gqa_groups = self.num_query_heads // max(self.num_kv_heads, 1)
        self.compute_dtype = compute_dtype
        self.q_proj = nn.Linear(dim, self.num_query_heads * self.head_dim, bias=False, device=device)
        self.k_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False, device=device)
        self.v_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False, device=device)
        self.o_proj = nn.Linear(self.num_query_heads * self.head_dim, dim, bias=False, device=device)
        eps = config.model.normalization_layer_epsilon
        self.q_norm = nn.RMSNorm(self.head_dim, eps=eps, dtype=torch.float32, device=device)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=eps, dtype=torch.float32, device=device)
        
        # Pre-calculate RoPE for the full context + small buffer
        rope_len = config.runtime.max_context_steps + 256
        self.rotary = RotaryEmbedding(
            self.head_dim,
            config.model.rope_min_timescale,
            config.model.rope_max_timescale,
            max_seq_len=rope_len,
            device=device,
        )

    def forward_incremental(
        self,
        x: torch.Tensor,
        pos: Optional[torch.Tensor],
        cache_slot,
    ) -> Tuple[torch.Tensor, object]:
        B, T, _ = x.shape
        if T != 1:
            raise ValueError("Attention expects sequence length 1 during decoding")
        orig_dtype = x.dtype
        q_proj = self._project_heads(self.q_proj, x, self.num_query_heads)
        k_proj = self._project_heads(self.k_proj, x, self.num_kv_heads)
        v_proj = self._project_heads(self.v_proj, x, self.num_kv_heads)
        q_proj = self.q_norm(q_proj)
        k_proj = self.k_norm(k_proj)
        if pos is not None:
            q_proj = self.rotary(q_proj, pos)
            k_proj = self.rotary(k_proj, pos)
        q = q_proj.transpose(1, 2)
        k = k_proj.transpose(1, 2)
        v = v_proj.transpose(1, 2)
        if cache_slot is not None:
            k_cache, v_cache, attn_mask = cache_slot.write_and_view(k, v)
        else:
            k_cache, v_cache = k, v
            attn_mask = None
        attn = F.scaled_dot_product_attention(
            q,
            k_cache,
            v_cache,
            scale=1.0,
            attn_mask=attn_mask,
            enable_gqa=self.num_gqa_groups > 1,
        )
        attn = attn.transpose(1, 2).contiguous()
        flat = attn.reshape(B, T, self.num_query_heads * self.head_dim)
        out = self.o_proj(flat.to(torch.float32))
        return out.to(orig_dtype), cache_slot

    def _project_heads(self, layer: nn.Linear, x: torch.Tensor, heads: int) -> torch.Tensor:
        proj = layer(x.to(torch.float32))
        B, T, _ = proj.shape
        proj = proj.view(B, T, heads, self.head_dim)
        return proj.to(self.compute_dtype)

    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor],
        cache=None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self.forward_incremental(x, positions, cache)



class MultiStreamEmbedding(nn.Module):
    """Port of dia_v2 MultiStreamEmbed."""

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        pad_id: int,
        *,
        output_dtype: torch.dtype,
        low_rank_dim: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.dtype = output_dtype
        base_dim = low_rank_dim if low_rank_dim is not None else dim
        self.embedding = nn.Embedding(vocab_size, base_dim, device=device)
        self.main_proj = nn.Linear(base_dim, dim, bias=False, device=device)
        self.second_proj = nn.Linear(base_dim, dim, bias=False, device=device)


    def forward(self, main_inputs: torch.Tensor, second_inputs: torch.Tensor) -> torch.Tensor:
        main_inputs = main_inputs.long()
        second_inputs = second_inputs.long()
        if self.pad_id is not None:
            second_is_pad = second_inputs == self.pad_id
        else:
            second_is_pad = torch.zeros_like(second_inputs, dtype=torch.bool)
        use_second = ~second_is_pad
        emb_main = self.embedding(main_inputs)
        emb_second = self.embedding(second_inputs)
        out_main = self.main_proj(emb_main.to(torch.float32))
        out_second = self.second_proj(emb_second.to(torch.float32))
        zeros = torch.zeros_like(out_second)
        y = out_main + torch.where(use_second.unsqueeze(-1), out_second, zeros)
        target_dtype = self.dtype if self.dtype is not None else y.dtype
        return y.to(target_dtype)


class Mlp(nn.Module):
    """Port of dia_v2 MlpBlock (two-activation gated MLP)."""

    def __init__(
        self,
        dim: int,
        hidden: int,
        compute_dtype: torch.dtype,
        activations: Sequence[str],
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if len(activations) != 2:
            raise ValueError("Mlp expects two activation functions.")
        self.dtype = compute_dtype
        self.hidden = hidden
        self.branch_count = len(activations)
        self.wi = nn.Linear(dim, self.branch_count * hidden, bias=False, device=device)
        self.wo = nn.Linear(hidden, dim, bias=False, device=device)
        self.activation_fns = [_get_activation(activations[0]), _get_activation(activations[1])]


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = self.wi(x.to(torch.float32))
        proj = proj.view(*x.shape[:-1], self.branch_count, self.hidden).to(self.dtype)
        gate, up = proj.unbind(dim=-2)
        hidden = self.activation_fns[0](gate) * self.activation_fns[1](up)
        out = self.wo(hidden.to(torch.float32))
        return out.to(self.dtype)
