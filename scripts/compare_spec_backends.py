#!/usr/bin/env python
"""Compare spectrogram outputs across backends using extract_spectrograms."""
from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from allin1_mlx.spectrogram import extract_spectrograms


def _iter_tracks(demix_dir: Path, limit: int | None):
  root = demix_dir
  if (demix_dir / "htdemucs").is_dir():
    root = demix_dir / "htdemucs"
  tracks = sorted([p for p in root.iterdir() if p.is_dir()])
  if limit is not None:
    tracks = tracks[:limit]
  return tracks


def _metrics(a: np.ndarray, b: np.ndarray):
  if a.shape != b.shape:
    min_frames = min(a.shape[1], b.shape[1])
    min_bins = min(a.shape[2], b.shape[2])
    a = a[:, :min_frames, :min_bins]
    b = b[:, :min_frames, :min_bins]
  diff = a - b
  return {
    "mean_abs": float(np.mean(np.abs(diff))),
    "max_abs": float(np.max(np.abs(diff))),
    "rmse": float(np.sqrt(np.mean(diff ** 2))),
    "mean_rel": float(np.mean(np.abs(diff) / (np.abs(a) + 1e-8))),
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("demix_dir", type=Path, help="Path to htdemucs output dir")
  parser.add_argument("--backend-a", type=str, default="madmom", choices=["madmom", "torch", "mlx"])
  parser.add_argument("--backend-b", type=str, default="torch", choices=["madmom", "torch", "mlx"])
  parser.add_argument("--limit", type=int, default=None)
  parser.add_argument("--torch-device", type=str, default="cpu")
  parser.add_argument("--torch-dtype", type=str, default="float32")
  args = parser.parse_args()

  tracks = _iter_tracks(args.demix_dir, args.limit)
  if not tracks:
    raise SystemExit("No track directories found under demix_dir.")

  with TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    out_a = tmp_path / "spec_a"
    out_b = tmp_path / "spec_b"

    paths_a = extract_spectrograms(
      tracks,
      out_a,
      multiprocess=False,
      backend=args.backend_a,
      torch_device=args.torch_device,
      torch_dtype=args.torch_dtype,
    )
    paths_b = extract_spectrograms(
      tracks,
      out_b,
      multiprocess=False,
      backend=args.backend_b,
      torch_device=args.torch_device,
      torch_dtype=args.torch_dtype,
    )

    all_metrics = []
    for path_a, path_b in zip(paths_a, paths_b):
      spec_a = np.load(str(path_a))
      spec_b = np.load(str(path_b))
      metrics = _metrics(spec_a, spec_b)
      all_metrics.append(metrics)
      print(
        f"{path_a.stem}: "
        f"mean_abs={metrics['mean_abs']:.6f} "
        f"max_abs={metrics['max_abs']:.6f} "
        f"rmse={metrics['rmse']:.6f} "
        f"mean_rel={metrics['mean_rel']:.6f}"
      )

    if all_metrics:
      mean = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0].keys()}
      print(
        "overall: "
        f"mean_abs={mean['mean_abs']:.6f} "
        f"max_abs={mean['max_abs']:.6f} "
        f"rmse={mean['rmse']:.6f} "
        f"mean_rel={mean['mean_rel']:.6f}"
      )


if __name__ == "__main__":
  main()
