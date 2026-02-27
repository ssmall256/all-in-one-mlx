#!/usr/bin/env python3
"""
Convert all-in-one PyTorch checkpoints to MLX-compatible weights.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from allin1_mlx.models.mlx_conversion import (
  _load_torch_checkpoint,
  convert_state_dict,
  convert_torch_checkpoint_to_mlx,
  summarize_converted_shapes,
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser("convert_allinone_weights_to_mlx")
  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument("--model", type=str, help="Pretrained model name (e.g. harmonix-fold0)")
  group.add_argument("--checkpoint", type=Path, help="Path to local PyTorch checkpoint (.pth)")
  parser.add_argument("--cache-dir", type=Path, default=None, help="HF cache dir for pretrained weights")
  parser.add_argument("--output", type=Path, default=None, help="Output path (.safetensors or .npz)")
  parser.add_argument("--no-config", action="store_true", help="Do not write config sidecar")
  return parser.parse_args()


def main() -> None:
  args = parse_args()

  torch_checkpoint = args.checkpoint
  pt_state = None
  if torch_checkpoint is not None:
    pt_state, _ = _load_torch_checkpoint(torch_checkpoint)

  out_path, config_path = convert_torch_checkpoint_to_mlx(
    model_name=args.model,
    torch_checkpoint=torch_checkpoint,
    cache_dir=args.cache_dir,
    output=args.output,
    save_config=not args.no_config,
  )

  if pt_state is not None:
    mlx_state = convert_state_dict(pt_state)
    total, transposed = summarize_converted_shapes(pt_state, mlx_state)
    print("Conversion complete")
    print(f" - Parameters processed: {total} - Layout-adjusted: {transposed}")

  print(f" - Written weights: {out_path}")
  if config_path is not None:
    print(f" - Written config:  {config_path}")


if __name__ == "__main__":
  main()
