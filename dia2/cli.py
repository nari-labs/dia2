from __future__ import annotations

import argparse

import torch

from .engine import Dia2
from .generation import (
    build_generation_config,
    load_script_text,
    validate_generation_params,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate audio with Dia2")
    parser.add_argument("--config", help="Path to config.json (overrides repo lookup)")
    parser.add_argument(
        "--weights", help="Path to model.safetensors (overrides repo lookup)"
    )
    parser.add_argument(
        "--hf",
        required=False,
        help="Hugging Face repo id to download config/weights from (e.g. nari-labs/Dia2-2B)",
    )
    parser.add_argument(
        "--input", default="input.txt", help="Script text file (default: input.txt)"
    )
    parser.add_argument("output", help="Output WAV path")
    parser.add_argument(
        "--device",
        default=None,
        help="Computation device (defaults to cuda if available, else cpu)",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "bfloat16"],
        default="bfloat16",
        help="Computation dtype (default: bfloat16)",
    )
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--tokenizer", help="Tokenizer repo or local path override")
    parser.add_argument(
        "--mimi", help="Mimi repo id override (defaults to config/assets)"
    )
    parser.add_argument("--prefix-speaker-1", help="Prefix audio file for speaker 1")
    parser.add_argument("--prefix-speaker-2", help="Prefix audio file for speaker 2")
    parser.add_argument(
        "--include-prefix",
        action="store_true",
        help="Keep prefix audio in the final waveform (default: trimmed)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print generation progress logs"
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Run generation with CUDA graph capture",
    )
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Run generation with torch.compile optimizations",
    )
    args = parser.parse_args()

    device = args.device
    if device is None or device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = args.dtype or "bfloat16"

    repo = args.hf
    if repo:
        dia = Dia2(
            repo=repo,
            device=device,
            dtype=dtype,
            tokenizer_id=args.tokenizer,
            mimi_id=args.mimi,
        )
    elif args.config and args.weights:
        dia = Dia2.from_local(
            config_path=args.config,
            weights_path=args.weights,
            device=device,
            dtype=dtype,
            tokenizer_id=args.tokenizer,
            mimi_id=args.mimi,
        )
    else:
        raise ValueError("Provide --hf/--variant or both --config and --weights")

    script = load_script_text(args.input)
    temperature, top_k, cfg_scale = validate_generation_params(
        temperature=args.temperature,
        top_k=args.topk,
        cfg_scale=args.cfg,
    )
    config = build_generation_config(
        temperature=temperature,
        top_k=top_k,
        cfg_scale=cfg_scale,
    )
    overrides = {}
    if args.cuda_graph:
        overrides["use_cuda_graph"] = True
    if args.torch_compile:
        overrides["use_torch_compile"] = True
    if args.prefix_speaker_1:
        overrides["prefix_speaker_1"] = args.prefix_speaker_1
    if args.prefix_speaker_2:
        overrides["prefix_speaker_2"] = args.prefix_speaker_2
    if args.include_prefix:
        overrides["include_prefix"] = True

    dia.generate(
        script,
        config=config,
        output_wav=args.output,
        verbose=args.verbose,
        **overrides,
    )


if __name__ == "__main__":
    main()
