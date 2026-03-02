#!/usr/bin/env python3
"""Run beat/downbeat parity comparisons across upstream, mps, and mlx environments."""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _nearest_deltas(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
  if reference.size == 0 or candidate.size == 0:
    return np.array([], dtype=float)
  j = 0
  out: List[float] = []
  for t in reference:
    while j + 1 < candidate.size and abs(candidate[j + 1] - t) <= abs(candidate[j] - t):
      j += 1
    out.append(float(candidate[j] - t))
  return np.array(out, dtype=float)


def _delta_summary(reference: np.ndarray, candidate: np.ndarray) -> Dict[str, Any]:
  deltas = _nearest_deltas(reference, candidate)
  if deltas.size == 0:
    return {
      "count": 0,
      "median_signed": None,
      "mean_signed": None,
      "median_abs": None,
      "p90_abs": None,
      "max_abs": None,
    }
  abs_deltas = np.abs(deltas)
  return {
    "count": int(deltas.size),
    "median_signed": float(np.median(deltas)),
    "mean_signed": float(np.mean(deltas)),
    "median_abs": float(np.median(abs_deltas)),
    "p90_abs": float(np.quantile(abs_deltas, 0.9)),
    "max_abs": float(np.max(abs_deltas)),
  }


def _run_analysis(
  python_exe: Path,
  module_name: str,
  audio_path: Path,
  out_dir: Path,
  kwargs: Dict[str, Any],
  timeout: int,
) -> Dict[str, Any]:
  payload = {
    "module": module_name,
    "audio_path": str(audio_path),
    "out_dir": str(out_dir),
    "kwargs": kwargs,
  }
  code = r"""
import importlib
import inspect
import json
import sys
from pathlib import Path

payload = json.loads(sys.argv[1])
module = importlib.import_module(payload["module"])
fn = getattr(module, "analyze")
sig = inspect.signature(fn)

kwargs = dict(payload["kwargs"])
kwargs["out_dir"] = payload["out_dir"]
filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}

result = fn(payload["audio_path"], **filtered)
if isinstance(result, list):
  result = result[0]

beats = [float(v) for v in (getattr(result, "beats", None) or [])]
downbeats = [float(v) for v in (getattr(result, "downbeats", None) or [])]
print(json.dumps({
  "bpm": getattr(result, "bpm", None),
  "beats": beats,
  "downbeats": downbeats,
  "beat_positions": list(getattr(result, "beat_positions", None) or []),
  "first_beat": beats[0] if beats else None,
  "last_beat": beats[-1] if beats else None,
  "first_downbeat": downbeats[0] if downbeats else None,
  "last_downbeat": downbeats[-1] if downbeats else None,
}, sort_keys=True))
"""
  proc = subprocess.run(
    [str(python_exe), "-c", code, json.dumps(payload)],
    check=True,
    capture_output=True,
    text=True,
    timeout=timeout,
  )
  lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
  if not lines:
    raise RuntimeError(f"No JSON output produced by {python_exe}")
  return json.loads(lines[-1])


def _comparison_row(
  ref_name: str,
  cand_name: str,
  ref: Dict[str, Any],
  cand: Dict[str, Any],
) -> Dict[str, Any]:
  ref_beats = np.array(ref.get("beats", []), dtype=float)
  cand_beats = np.array(cand.get("beats", []), dtype=float)
  ref_down = np.array(ref.get("downbeats", []), dtype=float)
  cand_down = np.array(cand.get("downbeats", []), dtype=float)
  return {
    "reference": ref_name,
    "candidate": cand_name,
    "beats": _delta_summary(ref_beats, cand_beats),
    "downbeats": _delta_summary(ref_down, cand_down),
    "beat_count_ref": int(ref_beats.size),
    "beat_count_candidate": int(cand_beats.size),
    "downbeat_count_ref": int(ref_down.size),
    "downbeat_count_candidate": int(cand_down.size),
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("audio_path", type=Path, help="Audio file to analyze.")
  parser.add_argument("--upstream-python", type=Path, required=True, help="Python executable for upstream all-in-one env.")
  parser.add_argument("--mps-python", type=Path, required=True, help="Python executable for all-in-one-mps env.")
  parser.add_argument("--mlx-python", type=Path, required=True, help="Python executable for all-in-one-mlx env.")
  parser.add_argument("--mlx-weights-dir", type=Path, default=None, help="Optional MLX weights dir for all-in-one-mlx.")
  parser.add_argument("--timeout", type=int, default=1800, help="Timeout in seconds per run.")
  parser.add_argument("--output", type=Path, default=None, help="Write JSON report to this path.")
  args = parser.parse_args()

  with tempfile.TemporaryDirectory(prefix="allin1_beat_parity_") as tmp:
    tmp_dir = Path(tmp)
    runs: Dict[str, Dict[str, Any]] = {}
    run_matrix: List[Tuple[str, Path, str, Dict[str, Any]]] = [
      (
        "upstream_cpu",
        args.upstream_python,
        "allin1",
        {"device": "cpu", "multiprocess": False, "keep_byproducts": False, "overwrite": True},
      ),
      (
        "mps",
        args.mps_python,
        "allin1",
        {"device": "mps", "force_mps": True, "multiprocess": False, "keep_byproducts": False, "overwrite": True},
      ),
      (
        "mlx_keep_true_fast",
        args.mlx_python,
        "allin1_mlx",
        {"device": "mlx", "multiprocess": False, "keep_byproducts": True, "overwrite": "all", "spec_backend": "mlx_fast", "spec_fast_guard": False},
      ),
      (
        "mlx_keep_false_fast",
        args.mlx_python,
        "allin1_mlx",
        {"device": "mlx", "multiprocess": False, "keep_byproducts": False, "overwrite": "all", "spec_backend": "mlx_fast", "spec_fast_guard": False},
      ),
      (
        "mlx_keep_true_ref",
        args.mlx_python,
        "allin1_mlx",
        {"device": "mlx", "multiprocess": False, "keep_byproducts": True, "overwrite": "all", "spec_backend": "mlx", "spec_fast_guard": False},
      ),
      (
        "mlx_keep_false_ref",
        args.mlx_python,
        "allin1_mlx",
        {"device": "mlx", "multiprocess": False, "keep_byproducts": False, "overwrite": "all", "spec_backend": "mlx", "spec_fast_guard": False},
      ),
    ]

    if args.mlx_weights_dir is not None:
      for idx, (name, py, mod, kwargs) in enumerate(run_matrix):
        if mod == "allin1_mlx":
          kwargs = dict(kwargs)
          kwargs["mlx_weights_dir"] = str(args.mlx_weights_dir)
          run_matrix[idx] = (name, py, mod, kwargs)

    for run_name, python_exe, module_name, kwargs in run_matrix:
      out_dir = tmp_dir / run_name
      out_dir.mkdir(parents=True, exist_ok=True)
      runs[run_name] = _run_analysis(
        python_exe=python_exe,
        module_name=module_name,
        audio_path=args.audio_path,
        out_dir=out_dir,
        kwargs=kwargs,
        timeout=args.timeout,
      )

    comparisons = [
      _comparison_row("upstream_cpu", "mps", runs["upstream_cpu"], runs["mps"]),
      _comparison_row("upstream_cpu", "mlx_keep_true_fast", runs["upstream_cpu"], runs["mlx_keep_true_fast"]),
      _comparison_row("upstream_cpu", "mlx_keep_false_fast", runs["upstream_cpu"], runs["mlx_keep_false_fast"]),
      _comparison_row("upstream_cpu", "mlx_keep_true_ref", runs["upstream_cpu"], runs["mlx_keep_true_ref"]),
      _comparison_row("upstream_cpu", "mlx_keep_false_ref", runs["upstream_cpu"], runs["mlx_keep_false_ref"]),
    ]

    report = {
      "audio_path": str(args.audio_path),
      "runs": {
        name: {
          "bpm": run.get("bpm"),
          "beat_count": len(run.get("beats", [])),
          "downbeat_count": len(run.get("downbeats", [])),
          "first_beat": run.get("first_beat"),
          "last_beat": run.get("last_beat"),
          "first_downbeat": run.get("first_downbeat"),
          "last_downbeat": run.get("last_downbeat"),
        }
        for name, run in runs.items()
      },
      "comparisons": comparisons,
    }

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
      args.output.parent.mkdir(parents=True, exist_ok=True)
      args.output.write_text(text + "\n")


if __name__ == "__main__":
  main()
