from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch


@dataclass
class CacheSlot:
    keys: torch.Tensor
    values: torch.Tensor

    def __post_init__(self) -> None:
        self.max_steps = self.keys.shape[2]
        self.head_dim = self.keys.shape[3]
        self.flat_heads = self.keys.shape[0] * self.keys.shape[1]
        device = self.keys.device
        self.length = torch.zeros((), dtype=torch.long, device=device)
        self.positions = torch.arange(self.max_steps, dtype=torch.long, device=device)

    @classmethod
    def allocate(
        cls,
        *,
        batch_size: int,
        heads: int,
        max_steps: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "CacheSlot":
        keys = torch.zeros(batch_size, heads, max_steps, head_dim, device=device, dtype=dtype)
        values = torch.zeros_like(keys)
        return cls(keys, values)

    def reset(self) -> None:
        self.length.zero_()

    # Due to many CacheSlot instances being used in a model, we disable
    # compilation for this method to avoid excessive compile times.
    @torch.compiler.disable
    def write_and_view(
        self,
        key_chunk: torch.Tensor,
        value_chunk: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        step = key_chunk.shape[2]
        start = self.length
        indices = self.positions[:step] + start
        expanded = indices.unsqueeze(0).expand(self.flat_heads, -1)

        flat_keys = self.keys.view(self.flat_heads, self.max_steps, self.head_dim)
        flat_values = self.values.view(self.flat_heads, self.max_steps, self.head_dim)
        flat_key_chunk = key_chunk.reshape(self.flat_heads, step, self.head_dim)
        flat_value_chunk = value_chunk.reshape(self.flat_heads, step, self.head_dim)
        scatter_index = expanded.unsqueeze(-1).expand_as(flat_key_chunk)
        flat_keys.scatter_(1, scatter_index, flat_key_chunk)
        flat_values.scatter_(1, scatter_index, flat_value_chunk)

        self.length.add_(step)
        bool_mask = (self.positions >= self.length).view(1, 1, 1, self.max_steps)
        mask_dtype = self.keys.dtype
        mask_value = torch.finfo(mask_dtype).min
        attn_mask = torch.zeros_like(bool_mask, dtype=mask_dtype)
        attn_mask = attn_mask.masked_fill(bool_mask, mask_value)
        return self.keys, self.values, attn_mask


class KVCache:
    def __init__(self, slots: List[CacheSlot]) -> None:
        self.slots = slots

    @classmethod
    def allocate(
        cls,
        *,
        num_layers: int,
        batch_size: int,
        heads: int,
        max_steps: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "KVCache":
        slots = [
            CacheSlot.allocate(
                batch_size=batch_size,
                heads=heads,
                max_steps=max_steps,
                head_dim=head_dim,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ]
        return cls(slots)

    def get_slot(self, index: int) -> CacheSlot:
        return self.slots[index]

    def reset(self) -> None:
        for slot in self.slots:
            slot.reset()

    clear = reset


__all__ = ["CacheSlot", "KVCache"]
