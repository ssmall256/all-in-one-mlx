import time

import mlx.core as mx
import numpy as np

from ..config import Config
from ..typings import AllInOneOutput

# Use optimized native DBN implementation (3x faster with Cython)
from .dbn_native import DBNDownBeatTrackingProcessor

_DBN_CACHE = {}
_EPSILON = mx.array(1e-8)  # Cache epsilon constant


def postprocess_metrical_structure_mlx(
  logits: AllInOneOutput,
  cfg: Config,
  prob_beat: np.ndarray = None,
  prob_downbeat: np.ndarray = None,
  timings: dict = None,
):
  t0 = time.perf_counter()
  cache_key = (cfg.best_threshold_downbeat, cfg.fps)
  if cache_key not in _DBN_CACHE:
    _DBN_CACHE[cache_key] = DBNDownBeatTrackingProcessor(
      beats_per_bar=[3, 4],
      threshold=cfg.best_threshold_downbeat,
      fps=cfg.fps,
    )
  postprocessor_downbeat = _DBN_CACHE[cache_key]

  # Use pre-computed probabilities if provided, otherwise compute from logits
  if prob_beat is not None and prob_downbeat is not None:
    activations_beat = mx.array(prob_beat)
    activations_downbeat = mx.array(prob_downbeat)
  else:
    # Fused sigmoid operations
    activations_beat = mx.sigmoid(logits.logits_beat[0])
    activations_downbeat = mx.sigmoid(logits.logits_downbeat[0])

  # Compute all three channels in a fused manner
  # xbeat = max(eps, beat - downbeat)
  # no = (2 - beat - downbeat) / 2 = 1 - (beat + downbeat) / 2
  activations_xbeat = mx.maximum(_EPSILON, activations_beat - activations_downbeat)
  activations_no = 1.0 - (activations_beat + activations_downbeat) * 0.5

  # Stack and normalize in one operation
  activations_combined = mx.stack([activations_xbeat, activations_downbeat, activations_no], axis=-1)
  # Fused normalization
  norm_factor = mx.sum(activations_combined, axis=-1, keepdims=True)
  activations_combined = activations_combined / norm_factor

  # Force evaluation before converting to NumPy
  mx.eval(activations_combined)
  activations_combined = np.array(activations_combined)
  t1 = time.perf_counter()

  t2 = time.perf_counter()
  pred_downbeat_times = postprocessor_downbeat(activations_combined[:, :2])
  t3 = time.perf_counter()

  beats = pred_downbeat_times[:, 0]
  beat_positions = pred_downbeat_times[:, 1]
  downbeats = pred_downbeat_times[beat_positions == 1., 0]

  beats = beats.tolist()
  downbeats = downbeats.tolist()
  beat_positions = beat_positions.astype('int').tolist()

  if timings is not None:
    timings["metrical_prep"] = (t0, t1)
    timings["metrical_dbn"] = (t2, t3)

  return {
    'beats': beats,
    'downbeats': downbeats,
    'beat_positions': beat_positions,
  }
