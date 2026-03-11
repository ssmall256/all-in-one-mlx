#!/usr/bin/env python3
"""Thin wrapper around `mlx-weights convert` for all-in-one checkpoints."""
from __future__ import annotations

import argparse
from pathlib import Path

from mlx_weights import convert_model


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser("convert_allinone_weights_to_mlx")
  parser.add_argument("--model", required=True, type=str, help="Pretrained model name (e.g. harmonix-fold0)")
  parser.add_argument("--checkpoint", type=Path, help="Optional local PyTorch checkpoint (.pth)")
  parser.add_argument("--output", type=Path, default=None, help="Output safetensors path")
  parser.add_argument("--config", type=Path, default=None, help="Output config JSON path")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  output = args.output
  config = args.config
  if output is not None and config is None:
    config = output.with_suffix(".json")

  result = convert_model(
    f"allin1/{args.model}",
    input_path=args.checkpoint,
    output_path=output,
    extra_output_paths=None if config is None else {"config": config},
  )
  print(f"wrote {result.primary}")
  config_path = result.get("config")
  if config_path is not None:
    print(f"wrote {config_path}")


if __name__ == "__main__":
  main()
