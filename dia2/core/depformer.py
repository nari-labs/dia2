from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from ..config import DiaConfig
from .cache import KVCache
from .layers import MultiStreamEmbedding, Mlp, RotaryEmbedding
from .precision import Precision


class ScheduleAttention(nn.Module):
    """Depformer attention that mirrors dia_v2 ScheduleAttention."""

    def __init__(
        self,
        config: DiaConfig,
        compute_dtype: torch.dtype,
        device: Optional[torch.device] = None,
      ):
        super().__init__()
        dep_cfg = config.model.depformer
        runtime = config.runtime
        self.schedule = runtime.weights_schedule
        self.num_query_heads = dep_cfg.gqa_query_heads
        self.num_kv_heads = dep_cfg.kv_heads
        self.head_dim = dep_cfg.gqa_head_dim
        self.num_gqa_groups = self.num_query_heads // max(self.num_kv_heads, 1)
        self.apply_rope = dep_cfg.apply_rope
        self.used_ids = sorted(set(self.schedule))
        self.compute_dtype = compute_dtype

        self.in_proj = nn.ModuleDict(
            {
                str(i): nn.Linear(
                    dep_cfg.n_embd,
                    3 * self.num_query_heads * self.head_dim,
                    bias=False,
                    device=device,
                )
                for i in self.used_ids
            }
        )
        self.out_proj = nn.ModuleDict(
            {
                str(i): nn.Linear(
                    self.num_query_heads * self.head_dim,
                    dep_cfg.n_embd,
                    bias=False,
                    device=device,
                )
                for i in self.used_ids
            }
        )
        eps = config.model.normalization_layer_epsilon
        self.q_norm = nn.RMSNorm(self.head_dim, eps=eps, dtype=torch.float32, device=device)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=eps, dtype=torch.float32, device=device)

        if self.apply_rope:
            self.rotary = RotaryEmbedding(
                self.head_dim,
                config.model.rope_min_timescale,
                config.model.rope_max_timescale,
                device=device,
            )
            stage_count = max(len(self.schedule), 1)
            self.register_buffer(
                "stage_positions",
                torch.arange(stage_count, dtype=torch.long, device=device).view(stage_count, 1),
                persistent=False,
            )
        else:
            self.rotary = None
            self.register_buffer(
                "stage_positions",
                torch.zeros(0, 1, dtype=torch.long, device=device),
                persistent=False,
            )

    def _forward_incremental(
        self,
        x_t: torch.Tensor,
        in_proj: nn.Linear,
        pos_ids: Optional[torch.Tensor],
        cache_slot,
    ) -> Tuple[torch.Tensor, object]:
        bsz, seq, _ = x_t.shape
        proj = in_proj(x_t.to(torch.float32))
        proj = proj.view(bsz, seq, 3, self.num_query_heads, self.head_dim).to(self.compute_dtype)

        q_proj = self.q_norm(proj[:, :, 0])
        k_proj = self.k_norm(proj[:, :, 1])
        v_proj = proj[:, :, 2]

        if pos_ids is not None:
            q_proj = self.rotary(q_proj, pos_ids)
            k_proj = self.rotary(k_proj, pos_ids)

        q = q_proj.transpose(1, 2)
        k = k_proj.transpose(1, 2)
        v = v_proj.transpose(1, 2)

        if cache_slot is not None:
            k, v, attn_mask = cache_slot.write_and_view(k, v)
        else:
            attn_mask = None

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=1.0,
            attn_mask=attn_mask,
            enable_gqa=self.num_gqa_groups > 1,
        )
        attn = attn.transpose(1, 2).contiguous()
        flat = attn.reshape(bsz, seq, self.num_query_heads * self.head_dim)
        return flat, cache_slot

    # This method usually gets recompiled many times due to the `stage_index`
    # changing every step.
    @torch.compiler.disable(recursive=False)
    def forward_incremental(
        self,
        x_t: torch.Tensor,
        stage_index: int,
        cache_slot,
    ) -> Tuple[torch.Tensor, object]:
        pos_ids = None
        if self.apply_rope:
            pos_ids = self.stage_positions[stage_index : stage_index + 1]
            if pos_ids.device != x_t.device:
                pos_ids = pos_ids.to(x_t.device)
        module_index = self.schedule[stage_index]

        if torch.compiler.is_compiling():
          self._forward_incremental = torch.compile(
              self._forward_incremental,
              dynamic=True,
              mode="max-autotune-no-cudagraphs",
          )

        result = self._forward_incremental(
            x_t,
            in_proj=self.in_proj[str(module_index)],
            pos_ids=pos_ids,
            cache_slot=cache_slot,
        )
        flat, cache_slot = result
        out = self.out_proj[str(module_index)](flat.to(torch.float32))
        return out.to(x_t.dtype), cache_slot


class DepformerLayer(nn.Module):
    def __init__(
        self,
        config: DiaConfig,
        compute_dtype: torch.dtype,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        dep_cfg = config.model.depformer
        eps = config.model.normalization_layer_epsilon
        self.pre_norm = nn.RMSNorm(dep_cfg.n_embd, eps=eps, dtype=torch.float32, device=device)
        self.post_norm = nn.RMSNorm(dep_cfg.n_embd, eps=eps, dtype=torch.float32, device=device)
        self.self_attention = ScheduleAttention(config, compute_dtype, device=device)
        self.mlp = Mlp(
            dep_cfg.n_embd,
            dep_cfg.n_hidden,
            compute_dtype,
            tuple(config.model.depformer.mlp_activations),
            device=device,
        )

    def decode_step(
        self,
        x_t: torch.Tensor,
        stage_index: int,
        cache_slot,
    ) -> Tuple[torch.Tensor, object]:
        residual = x_t
        x_norm = self.pre_norm(x_t)
        sa_out, _ = self.self_attention.forward_incremental(x_norm, stage_index, cache_slot)
        x = residual + sa_out
        residual2 = x
        x_norm2 = self.post_norm(x)
        mlp_out = self.mlp(x_norm2)
        return residual2 + mlp_out, cache_slot


class Depformer(nn.Module):
    def __init__(
        self,
        config: DiaConfig,
        precision: Precision,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.config = config
        self.precision = precision
        dep_cfg = config.model.depformer
        data_cfg = config.data
        runtime = config.runtime

        self.num_audio_channels = max(0, data_cfg.channels - 2)
        self.num_depth = max(self.num_audio_channels - 1, 0)
        self.weights_schedule = runtime.weights_schedule

        self.audio_embeds = nn.ModuleList(
            [nn.Embedding(data_cfg.audio_vocab_size, dep_cfg.n_embd, device=device) for _ in range(self.num_depth)]
        )
        if dep_cfg.text_embedding:
            self.text_embed = MultiStreamEmbedding(
                data_cfg.text_vocab_size,
                dep_cfg.n_embd,
                pad_id=data_cfg.text_pad_token_id,
                output_dtype=precision.compute,
                device=device,
            )
        else:
            self.text_embed = None

        used_ids = sorted(set(self.weights_schedule))
        self.depformer_in = nn.ModuleDict(
            {
                str(i): nn.Linear(
                    config.model.decoder.n_embd,
                    dep_cfg.n_embd,
                    bias=False,
                    device=device,
                )
                for i in used_ids
            }
        )

        self.layers = nn.ModuleList([DepformerLayer(config, precision.compute, device=device) for _ in range(dep_cfg.n_layer)])
        self.norm = nn.RMSNorm(dep_cfg.n_embd, eps=config.model.normalization_layer_epsilon, device=device)
        self.logits_dtype = precision.logits
        self.logits = nn.ModuleList(
            [
                nn.Linear(dep_cfg.n_embd, data_cfg.audio_vocab_size, bias=False, device=device)
                for _ in range(self.num_depth)
            ]
        )
        self.audio_vocab_limit = min(data_cfg.audio_pad_token_id, data_cfg.audio_bos_token_id)

    def init_cache(self, batch_size: int, device: torch.device, max_steps: int) -> KVCache:
        heads = self.layers[0].self_attention.num_kv_heads
        head_dim = self.layers[0].self_attention.head_dim
        return KVCache.allocate(
            num_layers=len(self.layers),
            batch_size=batch_size,
            heads=heads,
            max_steps=max_steps,
            head_dim=head_dim,
            device=device,
            dtype=self.precision.compute,
        )

    def forward_step(
        self,
        prev_audio: torch.Tensor,
        transformer_out: torch.Tensor,
        stage_index: int,
        cache: KVCache,
        main_text: Optional[torch.Tensor],
        second_text: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, KVCache]:
        self._validate_inputs(stage_index, cache)
        return self._forward_stage(stage_index, prev_audio, transformer_out, cache, main_text, second_text)

    def _forward_stage(
        self,
        stage_index: int,
        prev_audio: torch.Tensor,
        transformer_out: torch.Tensor,
        cache: KVCache,
        main_text: Optional[torch.Tensor],
        second_text: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, KVCache]:
        prev_audio = prev_audio.long()
        weight_idx = self.weights_schedule[stage_index]
        token_emb = self.audio_embeds[stage_index](prev_audio[:, None]).to(self.precision.compute)
        if stage_index == 0 and self.text_embed is not None:
            if main_text is None or second_text is None:
                raise ValueError("stage 0 requires text tokens")
            token_emb = token_emb + self.text_embed(main_text[:, None], second_text[:, None])

        dep_in = self.depformer_in[str(weight_idx)](transformer_out.to(torch.float32))
        dep_in = dep_in.to(self.precision.compute)
        dep_in = dep_in + token_emb.to(dep_in.dtype)
        x = dep_in
        for idx, layer in enumerate(self.layers):
            slot = cache.get_slot(idx)
            x, _ = layer.decode_step(x, stage_index, slot)

        hidden = self.norm(x)
        logits = self.logits[stage_index](hidden.to(torch.float32))
        logits = logits.to(self.logits_dtype)
        logits = logits.unsqueeze(1)
        logits = logits[..., : self.audio_vocab_limit]
        return logits, cache

    def _validate_inputs(self, stage_index: int, cache: KVCache | None) -> None:
        if stage_index < 0 or stage_index >= self.num_depth:
            raise ValueError(f"stage_index {stage_index} out of range (depth={self.num_depth})")
        if cache is None:
            raise ValueError("depformer cache must be initialized")
