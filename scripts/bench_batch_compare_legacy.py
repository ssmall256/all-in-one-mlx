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
from allin1_mlx.postprocessing import estimate_tempo_from_beats
from allin1_mlx.postprocessing.functional_mlx import postprocess_functional_structure_mlx
from allin1_mlx.postprocessing.metrical_mlx import postprocess_metrical_structure_mlx
from allin1_mlx.typings import AllInOneOutput, AnalysisResult


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Compare optimized and legacy batch inference helpers.")
  parser.add_argument("--spec-path", type=Path, action="append", required=True, help="Input spectrogram .npy path")
  parser.add_argument("--warmup", type=int, default=1, help="Warmup rounds per mode")
  parser.add_argument("--iters", type=int, default=6, help="Measured rounds")
  parser.add_argument("--model", type=str, default="harmonix-all", help="Model name")
  parser.add_argument("--mlx-weights-dir", type=Path, default=None, help="Optional MLX weights dir")
  parser.add_argument("--mlx-weights-path", type=Path, default=None, help="Optional MLX weights path")
  parser.add_argument("--mlx-config-path", type=Path, default=None, help="Optional MLX config path")
  parser.add_argument("--output-json", type=Path, required=True, help="Where to write comparison JSON")
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


def segment_payload(segments) -> list[dict]:
  payload: list[dict] = []
  for seg in segments:
    start = seg["start"] if isinstance(seg, dict) else seg.start
    end = seg["end"] if isinstance(seg, dict) else seg.end
    label = seg["label"] if isinstance(seg, dict) else seg.label
    payload.append({"start": float(start), "end": float(end), "label": str(label)})
  return payload


def run_inference_mlx_batch_legacy(
  paths,
  specs,
  model,
  include_activations: bool,
  include_embeddings: bool,
  compile_forward: bool = False,
  spec_timings=None,
  timings_list=None,
):
  import mlx.core as mx

  if timings_list is None:
    timings_list = [{} for _ in paths]

  if spec_timings is not None:
    for timings, spec_timing in zip(timings_list, spec_timings):
      timings["spec_load"] = spec_timing

  spec_batch = np.stack(specs, axis=0)
  t1 = time.perf_counter()
  forward = model
  if compile_forward:
    compiled_attr = "_compiled_forward_with_embeddings" if include_embeddings else "_compiled_forward_no_embeddings"
    if not hasattr(model, compiled_attr):
      from functools import partial

      state = [model.state]
      if include_embeddings:
        @partial(mx.compile, inputs=state)
        def _forward(x):
          outputs = model(x, return_embeddings=True)
          return (
            outputs.logits_beat,
            outputs.logits_downbeat,
            outputs.logits_section,
            outputs.logits_function,
            outputs.embeddings,
          )
      else:
        @partial(mx.compile, inputs=state)
        def _forward(x):
          outputs = model(x, return_embeddings=False)
          return (
            outputs.logits_beat,
            outputs.logits_downbeat,
            outputs.logits_section,
            outputs.logits_function,
          )
      setattr(model, compiled_attr, _forward)
    forward = getattr(model, compiled_attr)
    if include_embeddings:
      logits_beat, logits_downbeat, logits_section, logits_function, embeddings = forward(mx.array(spec_batch))
    else:
      logits_beat, logits_downbeat, logits_section, logits_function = forward(mx.array(spec_batch))
      embeddings = None
    logits = AllInOneOutput(
      logits_beat=logits_beat,
      logits_downbeat=logits_downbeat,
      logits_section=logits_section,
      logits_function=logits_function,
      embeddings=embeddings,
    )
  else:
    logits = forward(mx.array(spec_batch), return_embeddings=include_embeddings)

  prob_beat_mx = mx.sigmoid(logits.logits_beat)
  prob_downbeat_mx = mx.sigmoid(logits.logits_downbeat)
  prob_section_mx = mx.sigmoid(logits.logits_section)
  prob_function_mx = mx.softmax(logits.logits_function, axis=1)

  to_eval = [prob_beat_mx, prob_downbeat_mx, prob_section_mx, prob_function_mx]
  if include_embeddings and logits.embeddings is not None:
    to_eval.append(logits.embeddings)
  mx.eval(*to_eval)
  t2 = time.perf_counter()
  prob_beat = np.array(prob_beat_mx, copy=False)
  prob_downbeat = np.array(prob_downbeat_mx, copy=False)
  prob_section = np.array(prob_section_mx, copy=False)
  prob_function = np.array(prob_function_mx, copy=False)

  for timings in timings_list:
    timings["nn"] = (t1, t2)

  results = []
  for idx, path in enumerate(paths):
    t_post_start = time.perf_counter()
    item_logits = AllInOneOutput(
      logits_beat=logits.logits_beat[idx : idx + 1],
      logits_downbeat=logits.logits_downbeat[idx : idx + 1],
      logits_section=logits.logits_section[idx : idx + 1],
      logits_function=logits.logits_function[idx : idx + 1],
      embeddings=(logits.embeddings[idx : idx + 1] if logits.embeddings is not None else None),
    )

    metrical_timings = {}
    functional_timings = {}
    metrical_structure = postprocess_metrical_structure_mlx(
      item_logits,
      model.cfg,
      prob_beat=prob_beat[idx],
      prob_downbeat=prob_downbeat[idx],
      prob_beat_mx=prob_beat_mx[idx],
      prob_downbeat_mx=prob_downbeat_mx[idx],
      timings=metrical_timings,
    )
    functional_structure = postprocess_functional_structure_mlx(
      item_logits,
      model.cfg,
      prob_sections=prob_section[idx],
      prob_functions=prob_function[idx],
      timings=functional_timings,
    )
    t_post_end = time.perf_counter()

    timings = timings_list[idx]
    timings["postprocess"] = (t_post_start, t_post_end)
    for key, value in metrical_timings.items():
      timings[key] = value
    for key, value in functional_timings.items():
      timings[key] = value

    bpm = estimate_tempo_from_beats(metrical_structure["beats"])
    result = AnalysisResult(path=path, bpm=bpm, segments=functional_structure, **metrical_structure)

    if include_activations or include_embeddings:
      to_eval = []
      if include_activations:
        activations = {
          "beat": prob_beat[idx],
          "downbeat": prob_downbeat[idx],
          "segment": prob_section[idx],
          "label": prob_function[idx],
        }
      if include_embeddings:
        embeddings = item_logits.embeddings[0]
        to_eval.append(embeddings)
      if to_eval:
        mx.eval(*to_eval)

    if include_activations:
      result.activations = {
        "beat": np.array(activations["beat"], copy=False),
        "downbeat": np.array(activations["downbeat"], copy=False),
        "segment": np.array(activations["segment"], copy=False),
        "label": np.array(activations["label"], copy=False),
      }

    if include_embeddings:
      result.embeddings = np.array(embeddings, copy=False)

    results.append(result)

  return results


def run_mode(label: str, runner, paths, specs, iters: int, warmup: int):
  for _ in range(warmup):
    runner(
      paths=paths,
      specs=specs,
      model=MODEL,
      include_activations=False,
      include_embeddings=False,
      compile_forward=True,
      timings_list=[{} for _ in paths],
    )

  wall_times = []
  nn_times = []
  post_times = []
  representative = None
  for _ in range(iters):
    timings_list = [{} for _ in paths]
    t0 = time.perf_counter()
    results = runner(
      paths=paths,
      specs=specs,
      model=MODEL,
      include_activations=False,
      include_embeddings=False,
      compile_forward=True,
      timings_list=timings_list,
    )
    t1 = time.perf_counter()
    wall_times.append(t1 - t0)
    nn_start, nn_end = timings_list[0]["nn"]
    post_start, post_end = timings_list[0]["postprocess"]
    nn_times.append(float(nn_end - nn_start))
    post_times.append(float(post_end - post_start))
    if representative is None:
      representative = results
  assert representative is not None
  return {
    "label": label,
    "wall_s": summarize(wall_times),
    "nn_s": summarize(nn_times),
    "postprocess_s": summarize(post_times),
    "representative": representative,
  }


def parity_summary(legacy_results, optimized_results):
  per_track = []
  all_exact = True
  for legacy, optimized in zip(legacy_results, optimized_results):
    track_equal = (
      np.array_equal(np.asarray(legacy.beats), np.asarray(optimized.beats))
      and np.array_equal(np.asarray(legacy.downbeats), np.asarray(optimized.downbeats))
      and segment_payload(legacy.segments) == segment_payload(optimized.segments)
      and float(legacy.bpm) == float(optimized.bpm)
    )
    all_exact = all_exact and track_equal
    per_track.append(
      {
        "path": str(legacy.path),
        "exact": bool(track_equal),
        "beats_len": len(legacy.beats),
        "segments_len": len(legacy.segments),
      }
    )
  return {"all_exact": bool(all_exact), "per_track": per_track}


MODEL = None


def main() -> None:
  global MODEL
  args = parse_args()
  set_env()

  paths = [Path(f"bench_{idx}.wav") for idx, _ in enumerate(args.spec_path)]
  specs = [np.load(path) for path in args.spec_path]

  MODEL = load_pretrained_model_mlx(
    model_name=args.model,
    weights_dir=args.mlx_weights_dir,
    weights_path=args.mlx_weights_path,
    config_path=args.mlx_config_path,
    ensemble_parallel=True,
  )

  legacy = run_mode("legacy", run_inference_mlx_batch_legacy, paths, specs, args.iters, args.warmup)
  optimized = run_mode("optimized", run_inference_mlx_batch, paths, specs, args.iters, args.warmup)
  parity = parity_summary(legacy["representative"], optimized["representative"])

  args.output_json.parent.mkdir(parents=True, exist_ok=True)
  payload = {
    "spec_paths": [str(path) for path in args.spec_path],
    "spec_shapes": [list(np.load(path, mmap_mode="r").shape) for path in args.spec_path],
    "batch_size": len(args.spec_path),
    "warmup": int(args.warmup),
    "iters": int(args.iters),
    "legacy": {
      "wall_s": legacy["wall_s"],
      "nn_s": legacy["nn_s"],
      "postprocess_s": legacy["postprocess_s"],
    },
    "optimized": {
      "wall_s": optimized["wall_s"],
      "nn_s": optimized["nn_s"],
      "postprocess_s": optimized["postprocess_s"],
    },
    "parity": parity,
  }
  args.output_json.write_text(json.dumps(payload, indent=2))

  print("## Batch Compare")
  print(f"**Batch size:** `{len(args.spec_path)}`")
  print("")
  print("| Mode | Wall Mean (s) | NN Mean (s) | Post Mean (s) |")
  print("|---|---:|---:|---:|")
  print(
    f"| Legacy | **{legacy['wall_s']['mean']:.6f}** | "
    f"{legacy['nn_s']['mean']:.6f} | {legacy['postprocess_s']['mean']:.6f} |"
  )
  print(
    f"| Optimized | **{optimized['wall_s']['mean']:.6f}** | "
    f"{optimized['nn_s']['mean']:.6f} | {optimized['postprocess_s']['mean']:.6f} |"
  )
  print("")
  print(f"> {'✅' if parity['all_exact'] else '❌'} Exact parity: `{parity['all_exact']}`")


if __name__ == "__main__":
  main()
