from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 0.8
    top_k: int = 50


def _default_text_sampling() -> SamplingConfig:
    return SamplingConfig(temperature=0.6, top_k=50)


def _default_audio_sampling() -> SamplingConfig:
    return SamplingConfig(temperature=0.8, top_k=50)


@dataclass(frozen=True)
class PrefixConfig:
    speaker_1: Optional[str] = None
    speaker_2: Optional[str] = None
    include_audio: bool = False


@dataclass(frozen=True)
class GenerationConfig:
    text: SamplingConfig = field(default_factory=_default_text_sampling)
    audio: SamplingConfig = field(default_factory=_default_audio_sampling)
    cfg_scale: float = 2.0
    cfg_filter_k: int = 50
    initial_padding: int = 2
    prefix: Optional["PrefixConfig"] = None
    use_cuda_graph: bool = False
    use_torch_compile: bool = False

@dataclass(frozen=True)
class GenerationResult:
    audio_tokens: torch.Tensor
    waveform: torch.Tensor
    sample_rate: int
    timestamps: List[Tuple[str, float]]


def normalize_script(script: str | Sequence[str]) -> str:
    if isinstance(script, str):
        return script.strip()
    return "\n".join(line.strip() for line in script)


def load_script_text(path: str | Path) -> str:
    if path == "-":
        return sys.stdin.read().strip()
    path_obj = Path(path)
    if path_obj.exists():
        return path_obj.read_text().strip()
    return str(path).strip()


def validate_generation_params(
    *,
    temperature: float,
    top_k: int,
    cfg_scale: float,
) -> tuple[float, int, float]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if cfg_scale <= 0:
        raise ValueError("cfg_scale must be positive")
    return temperature, top_k, cfg_scale


def build_generation_config(
    *,
    temperature: float,
    top_k: int,
    cfg_scale: float,
) -> GenerationConfig:
    sampling = SamplingConfig(temperature=temperature, top_k=top_k)
    return GenerationConfig(
        text=sampling,
        audio=sampling,
        cfg_scale=cfg_scale,
    )


def merge_generation_config(
    *,
    base: GenerationConfig,
    overrides: Mapping[str, object],
) -> GenerationConfig:
    clean_overrides = {k: v for k, v in overrides.items() if v is not None}
    text_temp = clean_overrides.pop("temp_text", None)
    text_topk = clean_overrides.pop("topk_text", None)
    audio_temp = clean_overrides.pop("temp_audio", None)
    audio_topk = clean_overrides.pop("topk_audio", None)
    prefix_speaker_1 = clean_overrides.pop("prefix_speaker_1", None)
    prefix_speaker_2 = clean_overrides.pop("prefix_speaker_2", None)
    include_prefix = clean_overrides.pop("include_prefix", None)

    text_sampling = base.text
    if text_temp is not None or text_topk is not None:
        text_sampling = SamplingConfig(
            temperature=text_temp if text_temp is not None else text_sampling.temperature,
            top_k=text_topk if text_topk is not None else text_sampling.top_k,
        )

    audio_sampling = base.audio
    if audio_temp is not None or audio_topk is not None:
        audio_sampling = SamplingConfig(
            temperature=audio_temp if audio_temp is not None else audio_sampling.temperature,
            top_k=audio_topk if audio_topk is not None else audio_sampling.top_k,
        )

    prefix_cfg = base.prefix
    if (
        prefix_speaker_1 is not None
        or prefix_speaker_2 is not None
        or include_prefix is not None
        or prefix_cfg is not None
    ):
        prefix_cfg = prefix_cfg or PrefixConfig()
        prefix_cfg = PrefixConfig(
            speaker_1=prefix_speaker_1 if prefix_speaker_1 is not None else prefix_cfg.speaker_1,
            speaker_2=prefix_speaker_2 if prefix_speaker_2 is not None else prefix_cfg.speaker_2,
            include_audio=include_prefix if include_prefix is not None else prefix_cfg.include_audio,
        )

    return GenerationConfig(
        text=text_sampling,
        audio=audio_sampling,
        cfg_scale=clean_overrides.pop("cfg_scale", base.cfg_scale),
        cfg_filter_k=clean_overrides.pop("cfg_filter_k", base.cfg_filter_k),
        initial_padding=clean_overrides.pop("initial_padding", base.initial_padding),
        prefix=prefix_cfg,
        use_cuda_graph=clean_overrides.pop("use_cuda_graph", base.use_cuda_graph),
        use_torch_compile=clean_overrides.pop("use_torch_compile", base.use_torch_compile),
    )


__all__ = [
    "SamplingConfig",
    "GenerationConfig",
    "GenerationResult",
    "PrefixConfig",
    "normalize_script",
    "load_script_text",
    "validate_generation_params",
    "build_generation_config",
    "merge_generation_config",
]
