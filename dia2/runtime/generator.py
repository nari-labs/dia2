from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from ..core.cache import KVCache
from ..core.model import DecodeState
from ..generation import GenerationConfig
from ..audio.grid import delay_frames, mask_audio_logits, undelay_frames
from .context import RuntimeContext
from .state_machine import State, TokenIds
from .guidance import apply_classifier_guidance, sample_audio_logits
from .sampler import sample_token
from .voice_clone import PrefixPlan
from .logger import RuntimeLogger

_GRAPH_CUBLAS_READY = False


def _ensure_graph_cublas_ready(device: torch.device) -> None:
    global _GRAPH_CUBLAS_READY
    if _GRAPH_CUBLAS_READY or device.type != "cuda":
        return
    tmp = torch.empty((1, 1), device=device, dtype=torch.float32)
    torch.matmul(tmp, tmp)
    torch.cuda.synchronize()
    _GRAPH_CUBLAS_READY = True
@dataclass
class GenerationState:
    decode: DecodeState
    step_tokens: torch.Tensor
    audio_buf: torch.Tensor

    def trim_audio(self, limit: int, pad_token: int, ungenerated: int) -> torch.Tensor:
        trimmed = self.audio_buf[:, :, :limit]
        pad = torch.full_like(trimmed, pad_token)
        trimmed = torch.where(trimmed == ungenerated, pad, trimmed)
        self.audio_buf = trimmed
        return trimmed

    @property
    def transformer_cache(self) -> KVCache:
        return self.decode.transformer

    @transformer_cache.setter
    def transformer_cache(self, cache: KVCache) -> None:
        self.decode.transformer = cache

    @property
    def depformer_cache(self) -> KVCache:
        return self.decode.depformer

    @depformer_cache.setter
    def depformer_cache(self, cache: KVCache) -> None:
        self.decode.depformer = cache

    def reset_dep_cache(self) -> None:
        self.decode.depformer.reset()


@dataclass
class NetworkBuffers:
    text: torch.Tensor
    cb0: torch.Tensor
    dep: list[torch.Tensor]


def _allocate_network_buffers(runtime: RuntimeContext, branches: int) -> NetworkBuffers:
    device = runtime.device
    logits_dtype = runtime.precision.logits
    data_cfg = runtime.config.data
    text_logits = torch.empty((branches, 1, data_cfg.action_vocab_size), dtype=logits_dtype, device=device)
    cb0_logits = torch.empty((branches, 1, data_cfg.audio_vocab_size), dtype=logits_dtype, device=device)
    dep_vocab = runtime.model.depformer.audio_vocab_limit or data_cfg.audio_vocab_size
    dep_logits = [
        torch.empty((branches, 1, 1, dep_vocab), dtype=logits_dtype, device=device)
        for _ in range(runtime.model.depformer.num_depth)
    ]
    return NetworkBuffers(text=text_logits, cb0=cb0_logits, dep=dep_logits)


def build_initial_state(
    runtime: RuntimeContext,
    *,
    prefix: PrefixPlan | None = None,
) -> GenerationState:
    dep_q = runtime.model.depformer.num_audio_channels
    channels = 2 + dep_q
    branches = 2
    token_ids = runtime.constants
    step_tokens = torch.full(
        (branches, channels, 1),
        token_ids.pad,
        dtype=torch.long,
        device=runtime.device,
    )
    step_tokens[0, 0, 0] = token_ids.bos
    step_tokens[0, 1, 0] = token_ids.pad
    step_tokens[1, 0, 0] = token_ids.zero
    step_tokens[1, 1, 0] = token_ids.pad
    prefix_len = 0
    if prefix is not None:
        delayed = delay_frames(prefix.aligned_tokens, runtime.audio_delays, token_ids.audio_pad)
        prefix_len = delayed.shape[1]
    limit = runtime.config.runtime.max_context_steps
    total_steps = max(limit + prefix_len + 1, limit)
    decode_state = runtime.model.init_state(branches, runtime.device, total_steps)
    audio_buf = torch.full(
        (branches, dep_q, total_steps),
        token_ids.ungenerated,
        dtype=torch.long,
        device=runtime.device,
    )
    if prefix is not None:
        delayed = delay_frames(prefix.aligned_tokens, runtime.audio_delays, token_ids.audio_pad).to(runtime.device)
        audio_buf[0, :, : delayed.shape[1]] = delayed
        if branches > 1:
            audio_buf[1:, :, : delayed.shape[1]] = delayed
    return GenerationState(decode_state, step_tokens, audio_buf)


def _fill_audio_channels(
    step_tokens: torch.Tensor,
    audio_buf: torch.Tensor,
    delays: torch.Tensor,
    step: int,
    bos_token: int,
) -> None:
    channels = delays.numel()
    if channels == 0:
        return
    target = step_tokens[:, 2 : 2 + channels, 0]
    if step < audio_buf.shape[-1]:
        target.copy_(audio_buf[:, :channels, step])
    else:
        target.fill_(bos_token)
    mask = delays > step
    if mask.any().item():
        target[:, mask] = bos_token


def _execute_transformer_step(
    step_tokens: torch.Tensor,
    positions_view: torch.Tensor,
    generation: GenerationState,
    transformer_step,
    buffers: NetworkBuffers,
) -> torch.Tensor:
    hidden_t, text_logits_t, cb0_logits_t, present = transformer_step(
        step_tokens,
        positions_view,
        generation.transformer_cache,
    )
    buffers.text.copy_(text_logits_t)
    buffers.cb0.copy_(cb0_logits_t)
    generation.transformer_cache = present
    return hidden_t


def _execute_depformer_stage(
    stage_index: int,
    prev_audio: torch.Tensor,
    hidden_t: torch.Tensor,
    generation: GenerationState,
    depformer_step,
    main_tokens: Optional[torch.Tensor],
    second_tokens: Optional[torch.Tensor],
    buffers: NetworkBuffers,
) -> None:
    logits_stage, dep_present = depformer_step(
        prev_audio=prev_audio,
        transformer_out=hidden_t,
        stage_index=stage_index,
        cache=generation.depformer_cache,
        main_text=main_tokens if stage_index == 0 else None,
        second_text=second_tokens if stage_index == 0 else None,
    )
    target = buffers.dep[stage_index]
    if logits_stage.shape != target.shape:
        raise RuntimeError(
            f"depformer logits shape mismatch: {logits_stage.shape} vs {target.shape}"
        )
    target.copy_(logits_stage)
    generation.depformer_cache = dep_present


def _execute_transformer_graph(
    runtime: RuntimeContext,
    step_tokens: torch.Tensor,
    positions_view: torch.Tensor,
    branches: int,
    generation: GenerationState,
    transformer_step,
    buffers: NetworkBuffers,
    transformer_capture: Optional[Tuple[torch.cuda.CUDAGraph, torch.Tensor]],
    dep_captures: Optional[list[dict]],
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    if transformer_capture is None:
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            hidden_ref = _execute_transformer_step(
                step_tokens,
                positions_view,
                generation,
                transformer_step,
                buffers,
            )
        transformer_capture = (graph, hidden_ref)
        if runtime.model.depformer.num_depth > 0:
            dep_captures = []
            for idx in range(runtime.model.depformer.num_depth):
                capture = {
                    "graph": torch.cuda.CUDAGraph(),
                    "captured": False,
                    "prev_audio": torch.empty((branches,), dtype=torch.long, device=runtime.device),
                    "main_tokens": torch.empty((branches,), dtype=torch.long, device=runtime.device) if idx == 0 else None,
                    "second_tokens": torch.empty((branches,), dtype=torch.long, device=runtime.device) if idx == 0 else None,
                }
                dep_captures.append(capture)
    else:
        transformer_capture[0].replay()
    return transformer_capture, dep_captures

def _execute_depformer_graph(
    stage: int,
    prev_audio: torch.Tensor,
    hidden_t: torch.Tensor,
    generation: GenerationState,
    depformer_step,
    main_tokens: Optional[torch.Tensor],
    aux_tokens: Optional[torch.Tensor],
    buffers: NetworkBuffers,
    capture: dict[str, torch.Tensor | torch.cuda.CUDAGraph | bool],
) -> dict[str, torch.Tensor | torch.cuda.CUDAGraph | bool]:
    capture["prev_audio"].copy_(prev_audio)
    if capture["main_tokens"] is not None and stage == 0:
        capture["main_tokens"].copy_(main_tokens)
        capture["second_tokens"].copy_(aux_tokens)
    if not capture["captured"]:
        torch.cuda.synchronize()
        with torch.cuda.graph(capture["graph"]):
            _execute_depformer_stage(
                stage_index=stage,
                prev_audio=capture["prev_audio"],
                hidden_t=hidden_t,
                generation=generation,
                depformer_step=depformer_step,
                main_tokens=capture["main_tokens"],
                second_tokens=capture["second_tokens"],
                buffers=buffers,
            )
        capture["captured"] = True
    else:
        capture["graph"].replay()

    return capture


def run_generation_loop(
    runtime: RuntimeContext,
    *,
    state: State,
    generation: GenerationState,
    config: GenerationConfig,
    start_step: int = 0,
    logger: RuntimeLogger | None = None,
) -> tuple[Optional[int], torch.Tensor]:
    step_tokens = generation.step_tokens
    audio_buf = generation.audio_buf
    branches = step_tokens.shape[0]
    max_context = runtime.config.runtime.max_context_steps
    if max_context <= 0:
        raise ValueError("Runtime configuration must specify a positive max_context_steps")
    positions = torch.empty(1, 1, dtype=torch.long, device=runtime.device)
    main_tokens = torch.empty(branches, dtype=torch.long, device=runtime.device)
    aux_tokens = torch.empty(branches, dtype=torch.long, device=runtime.device)
    cfg_active = config.cfg_scale != 1.0
    token_ids = runtime.constants
    delay_tensor = runtime.audio_delay_tensor
    max_delay = int(delay_tensor.max().item()) if delay_tensor.numel() else 0
    flush_tail = max_delay + getattr(runtime.machine, "max_padding", 0)
    first_word_frame: Optional[int] = None
    eos_cutoff: Optional[int] = None
    last_step = start_step - 1
    use_graph = config.use_cuda_graph and runtime.device.type == "cuda"
    use_torch_compile = config.use_torch_compile and runtime.device.type == "cuda"
    transformer_needs_compiling = use_torch_compile
    depformer_needs_compiling = [use_torch_compile] * runtime.model.depformer.num_depth
    if use_torch_compile:
        sample_token_fn = torch.compile(sample_token, dynamic=True, mode="max-autotune-no-cudagraphs")
    transformer_step = runtime.transformer_step
    depformer_step = runtime.depformer_step
    buffers = _allocate_network_buffers(runtime, branches)
    positions_view = positions.expand(branches, -1)
    transformer_capture = None
    dep_captures: list[dict] | None = None
    if use_graph:
        _ensure_graph_cublas_ready(runtime.device)
    processed_steps = 0
    report_interval = 12
    with torch.inference_mode():
        for offset in range(max_context):
            if use_torch_compile:
                torch.compiler.cudagraph_mark_step_begin()
            t = start_step + offset
            if eos_cutoff is not None and t >= eos_cutoff:
                break
            if t + 1 >= audio_buf.shape[-1]:
                break
            generation.reset_dep_cache()
            positions.fill_(t)
            _fill_audio_channels(step_tokens, audio_buf, delay_tensor, t, token_ids.audio_bos)
            if branches > 1:
                step_tokens[1:, 0, 0] = token_ids.zero
                step_tokens[1:, 1, 0] = token_ids.pad
            if transformer_needs_compiling or not use_graph:
                if transformer_needs_compiling:
                    # Must use -no-cudagraphs variant as we are manually using graphs too.
                    transformer_step = torch.compile(
                        runtime.transformer_step,
                        dynamic=True,
                        mode="max-autotune-no-cudagraphs",
                    )
                    transformer_needs_compiling = False
                hidden_t = _execute_transformer_step(
                    step_tokens,
                    positions_view,
                    generation,
                    transformer_step,
                    buffers,
                )
            else:
                transformer_capture, dep_captures = _execute_transformer_graph(
                    runtime=runtime,
                    step_tokens=step_tokens,
                    positions_view=positions_view,
                    branches=branches,
                    generation=generation,
                    transformer_step=transformer_step,
                    buffers=buffers,
                    transformer_capture=transformer_capture,
                    dep_captures=dep_captures,
                )
                hidden_t = transformer_capture[1]

            guided_text = apply_classifier_guidance(buffers.text, cfg_active, config.cfg_scale, config.cfg_filter_k)
            if guided_text.shape[0] > 1:
                guided_text = guided_text[:1]

            text_token = sample_token_fn(
                guided_text,
                temp=config.text.temperature,
                top_k=config.text.top_k,
            ).item()

            main_token, aux_token, _ = runtime.machine.process(t, state, text_token)
            second_token = aux_token if aux_token != -1 else token_ids.pad
            if first_word_frame is None and main_token == token_ids.new_word:
                first_word_frame = t - config.initial_padding
            step_tokens[:, 0, 0] = main_token
            step_tokens[:, 1, 0] = second_token

            guided_cb0 = apply_classifier_guidance(buffers.cb0, cfg_active, config.cfg_scale, config.cfg_filter_k)
            if guided_cb0.shape[0] > 1:
                guided_cb0 = guided_cb0[:1]
            masked_cb0 = mask_audio_logits(guided_cb0, token_ids.audio_pad, token_ids.audio_bos)
            codebook_token = sample_audio_logits(masked_cb0, config.audio.temperature, config.audio.top_k)
            audio_buf[:, 0, t + 1] = codebook_token

            prev_audio = codebook_token.expand(branches)
            main_tokens.fill_(main_token)
            aux_tokens.fill_(second_token)
            for stage in range(runtime.model.depformer.num_depth):
                if use_graph and dep_captures is not None:
                    if depformer_needs_compiling[stage]:
                        runtime.model.depformer._forward_stage = torch.compile(
                            runtime.model.depformer._forward_stage,
                            dynamic=True,
                            mode="max-autotune-no-cudagraphs",
                        )
                        depformer_needs_compiling[stage] = False
                        _execute_depformer_stage(
                            stage_index=stage,
                            prev_audio=prev_audio,
                            hidden_t=hidden_t,
                            generation=generation,
                            depformer_step=depformer_step,
                            main_tokens=main_tokens,
                            second_tokens=aux_tokens,
                            buffers=buffers,
                        )
                    else:
                        dep_captures[stage] = _execute_depformer_graph(
                            stage=stage,
                            prev_audio=prev_audio,
                            hidden_t=hidden_t,
                            generation=generation,
                            depformer_step=depformer_step,
                            main_tokens=main_tokens,
                            aux_tokens=aux_tokens,
                            buffers=buffers,
                            capture=dep_captures[stage],
                        )

                else:
                    _execute_depformer_stage(
                        stage_index=stage,
                        prev_audio=prev_audio,
                        hidden_t=hidden_t,
                        generation=generation,
                        depformer_step=depformer_step,
                        main_tokens=main_tokens,
                        second_tokens=aux_tokens,
                        buffers=buffers,
                    )
                dep_logits = apply_classifier_guidance(buffers.dep[stage], cfg_active, config.cfg_scale, config.cfg_filter_k)
                if dep_logits.shape[0] > 1:
                    dep_logits = dep_logits[:1]
                stage_token = sample_audio_logits(
                    dep_logits,
                    config.audio.temperature,
                    config.audio.top_k,
                )
                audio_buf[:, stage + 1, t + 1] = stage_token
                prev_audio = stage_token.expand(branches)
            last_step = t
            if eos_cutoff is None and state.end_step is not None:
                eos_cutoff = state.end_step + flush_tail
            processed_steps = offset + 1
            if logger and processed_steps % report_interval == 0:
                logger.progress(processed_steps, max_context)

    if logger and processed_steps and processed_steps % report_interval != 0:
        logger.progress(processed_steps, max_context)

    if first_word_frame is None:
        first_word_frame = start_step
    if last_step < start_step:
        limit = min(start_step + 1, audio_buf.shape[-1])
    else:
        limit = min(last_step + 2, audio_buf.shape[-1])
    trimmed = generation.trim_audio(limit, token_ids.audio_pad, token_ids.ungenerated)
    return first_word_frame, trimmed


def decode_audio(runtime: RuntimeContext, tokens: torch.Tensor) -> torch.Tensor:
    if tokens.shape[-1] == 0:
        return torch.zeros(0, device=runtime.device)
    with torch.inference_mode():
        pcm = runtime.mimi.decode(tokens.to(runtime.device))
        return pcm[0, 0]

def warmup_with_prefix(
    runtime: RuntimeContext,
    plan: PrefixPlan,
    state: State,
    generation: GenerationState,
) -> int:
    step_tokens = generation.step_tokens
    model_state = generation.decode
    branches = step_tokens.shape[0]
    device = runtime.device
    tokens = plan.aligned_tokens.to(device)
    new_word_steps = set(plan.new_word_steps)
    positions = torch.empty(1, 1, dtype=torch.long, device=device)

    with torch.inference_mode():
        for t in range(plan.aligned_frames):
            positions.fill_(t)
            channels = tokens.shape[0]
            for cb in range(channels):
                delay = runtime.audio_delays[cb] if cb < len(runtime.audio_delays) else 0
                idx = t - delay
                value = tokens[cb, idx] if idx >= 0 else runtime.constants.audio_bos
                step_tokens[:, 2 + cb, 0] = value
            hidden, text_logits, cb0_logits, present = runtime.model.transformer.forward_step(
                step_tokens,
                positions.expand(branches, -1),
                model_state.transformer,
            )
            model_state.transformer = present

            forced = runtime.constants.new_word if t in new_word_steps else runtime.constants.pad
            main_token, aux_token, _ = runtime.machine.process(t, state, forced, is_forced=True)
            second_token = runtime.constants.pad if aux_token == -1 else aux_token
            step_tokens[0, 0, 0] = main_token
            step_tokens[0, 1, 0] = second_token
            if branches > 1:
                step_tokens[1:, 0, 0] = runtime.constants.zero
                step_tokens[1:, 1, 0] = runtime.constants.pad

    return max(plan.aligned_frames - 1, 0)
__all__ = [
    "build_initial_state",
    "run_generation_loop",
    "decode_audio",
    "warmup_with_prefix",
    "GenerationState",
]
