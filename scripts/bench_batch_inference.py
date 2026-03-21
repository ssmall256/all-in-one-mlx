#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np

from allin1_mlx.helpers import run_inference_mlx_batch
from allin1_mlx.models import load_pretrained_model_mlx


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Benchmark batched MLX inference on a fixed spectrogram.")
  parser.add_argument("--spec-path", type=Path, required=True, help="Path to an input spectrogram .npy file")
  parser.add_argument("--batch-size", type=int, default=4, help="Repeated batch size")
  parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations")
  parser.add_argument("--iters", type=int, default=8, help="Measured iterations")
  parser.add_argument("--model", type=str, default="harmonix-all", help="Model name")
  parser.add_argument("--mlx-weights-dir", type=Path, default=None, help="Optional MLX weights dir")
  parser.add_argument("--mlx-weights-path", type=Path, default=None, help="Optional MLX weights path")
  parser.add_argument("--mlx-config-path", type=Path, default=None, help="Optional MLX config path")
  parser.add_argument("--output-json", type=Path, required=True, help="Where to write benchmark summary JSON")
  parser.add_argument("--output-npz", type=Path, required=True, help="Where to write representative output arrays")
  return parser.parse_args()


def set_env() -> None:
  os.environ.setdefault("NATTEN_MLX", "1")
  os.environ.setdefault("NATTEN_MLX_BACKEND", "metal")
  os.environ.setdefault("NATTEN_MLX_COMPILE", "1")


def summarize(values: list[float]) -> dict[str, float]:
  arr = sorted(float(v) for v in values)
  return {
    "mean": float(statistics.fmean(arr)),
    "min": float(arr[0]),
    "median": float(statistics.median(arr)),
    "max": float(arr[-1]),
  }


def segment_payload(segments: list[dict]) -> list[dict]:
  payload: list[dict] = []
  for seg in segments:
    start = seg["start"] if isinstance(seg, dict) else seg.start
    end = seg["end"] if isinstance(seg, dict) else seg.end
    label = seg["label"] if isinstance(seg, dict) else seg.label
    payload.append(
      {
        "start": float(start),
        "end": float(end),
        "label": str(label),
      }
    )
  return payload


def main() -> None:
  args = parse_args()
  set_env()

  spec = np.load(args.spec_path)
  specs = [spec for _ in range(args.batch_size)]
  paths = [Path(f"bench_{idx}.wav") for idx in range(args.batch_size)]

  model = load_pretrained_model_mlx(
    model_name=args.model,
    weights_dir=args.mlx_weights_dir,
    weights_path=args.mlx_weights_path,
    config_path=args.mlx_config_path,
    ensemble_parallel=True,
  )

  def run_once():
    timings_list = [{} for _ in paths]
    t0 = time.perf_counter()
    results = run_inference_mlx_batch(
      paths=paths,
      specs=specs,
      model=model,
      include_activations=True,
      include_embeddings=True,
      compile_forward=True,
      timings_list=timings_list,
    )
    t1 = time.perf_counter()
    return results, timings_list, t1 - t0

  for _ in range(args.warmup):
    run_once()

  wall_times: list[float] = []
  nn_times: list[float] = []
  post_times: list[float] = []
  rep_result = None
  for _ in range(args.iters):
    results, timings_list, wall = run_once()
    wall_times.append(wall)
    nn_start, nn_end = timings_list[0]["nn"]
    post_start, post_end = timings_list[0]["postprocess"]
    nn_times.append(float(nn_end - nn_start))
    post_times.append(float(post_end - post_start))
    if rep_result is None:
      rep_result = results[0]

  assert rep_result is not None
  args.output_json.parent.mkdir(parents=True, exist_ok=True)
  args.output_npz.parent.mkdir(parents=True, exist_ok=True)

  summary = {
    "spec_path": str(args.spec_path),
    "spec_shape": list(spec.shape),
    "batch_size": int(args.batch_size),
    "warmup": int(args.warmup),
    "iters": int(args.iters),
    "wall_s": summarize(wall_times),
    "nn_s": summarize(nn_times),
    "postprocess_s": summarize(post_times),
    "beats_len": int(len(rep_result.beats)),
    "downbeats_len": int(len(rep_result.downbeats)),
    "segments_len": int(len(rep_result.segments)),
    "bpm": float(rep_result.bpm),
    "segments": segment_payload(rep_result.segments),
  }
  args.output_json.write_text(json.dumps(summary, indent=2))

  np.savez_compressed(
    args.output_npz,
    beats=np.asarray(rep_result.beats, dtype=np.float64),
    downbeats=np.asarray(rep_result.downbeats, dtype=np.float64),
    activ_beat=np.asarray(rep_result.activations["beat"], dtype=np.float32),
    activ_downbeat=np.asarray(rep_result.activations["downbeat"], dtype=np.float32),
    activ_segment=np.asarray(rep_result.activations["segment"], dtype=np.float32),
    activ_label=np.asarray(rep_result.activations["label"], dtype=np.float32),
    embeddings=np.asarray(rep_result.embeddings, dtype=np.float32),
  )

  print("## Batch Inference Benchmark")
  print(f"**Spec:** `{args.spec_path}`")
  print(f"**Shape:** `{tuple(spec.shape)}`")
  print(f"**Batch size:** `{args.batch_size}`")
  print("")
  print("| Metric | Mean | Median | Min | Max |")
  print("|---|---:|---:|---:|---:|")
  for label, stats in (
    ("Wall (s)", summary["wall_s"]),
    ("NN (s)", summary["nn_s"]),
    ("Postprocess (s)", summary["postprocess_s"]),
  ):
    print(
      f"| {label} | **{stats['mean']:.6f}** | {stats['median']:.6f} | "
      f"{stats['min']:.6f} | {stats['max']:.6f} |"
    )
  print("")
  print("> ✅ Representative output saved for parity comparison.")


if __name__ == "__main__":
  main()
