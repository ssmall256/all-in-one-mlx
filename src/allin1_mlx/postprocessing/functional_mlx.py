import time

import mlx.core as mx
import numpy as np

from ..config import HARMONIX_LABELS, Config
from ..typings import AllInOneOutput, Segment
from .helpers import event_frames_to_time, local_maxima_numpy_window, peak_picking


def postprocess_functional_structure_mlx(
  logits: AllInOneOutput,
  cfg: Config,
  prob_sections: np.ndarray = None,
  prob_functions: np.ndarray = None,
  timings: dict = None,
):
  t0 = time.perf_counter()

  # Use pre-computed probabilities if provided, otherwise compute from logits
  if prob_sections is None:
    raw_prob_sections = mx.sigmoid(logits.logits_section[0])
    prob_sections = np.array(raw_prob_sections)
    t1 = time.perf_counter()
    prob_sections, _ = local_maxima_numpy_window(
      prob_sections,
      filter_size=4 * cfg.min_hops_per_beat + 1,
    )
  else:
    t1 = time.perf_counter()

  if prob_functions is None:
    raw_prob_functions = mx.softmax(logits.logits_function[0], axis=0)
    prob_functions = np.array(raw_prob_functions)
  t2 = time.perf_counter()

  boundary_candidates = peak_picking(
    boundary_activation=prob_sections,
    window_past=12 * cfg.fps,
    window_future=12 * cfg.fps,
  )
  boundary = boundary_candidates > 0.0

  duration = len(prob_sections) * cfg.hop_size / cfg.sample_rate
  pred_boundary_times = event_frames_to_time(boundary, cfg)
  if len(pred_boundary_times) == 0:
    pred_boundary_times = np.array([0.0, duration], dtype=float)
  else:
    if pred_boundary_times[0] != 0:
      pred_boundary_times = np.insert(pred_boundary_times, 0, 0)
    if pred_boundary_times[-1] != duration:
      pred_boundary_times = np.append(pred_boundary_times, duration)
  pred_boundaries = np.stack([pred_boundary_times[:-1], pred_boundary_times[1:]]).T

  pred_boundary_indices = np.flatnonzero(boundary)
  pred_boundary_indices = pred_boundary_indices[pred_boundary_indices > 0]
  prob_segment_function = np.split(prob_functions, pred_boundary_indices, axis=1)
  pred_labels = [p.mean(axis=1).argmax().item() for p in prob_segment_function]
  t3 = time.perf_counter()

  segments = []
  for (start, end), label in zip(pred_boundaries, pred_labels):
    segment = Segment(
      start=start,
      end=end,
      label=HARMONIX_LABELS[label],
    )
    segments.append(segment)

  if timings is not None:
    timings["functional_probs"] = (t0, t1)
    timings["functional_local_maxima"] = (t1, t2)
    timings["functional_boundaries"] = (t2, t3)

  return segments
