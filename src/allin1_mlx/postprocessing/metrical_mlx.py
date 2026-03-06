import time

import mlx.core as mx
import numpy as np

from ..config import Config
from ..typings import AllInOneOutput

# Use optimized native DBN implementation (3x faster with Cython)
from .dbn_native import DBNDownBeatTrackingProcessor

_DBN_CACHE = {}
_EPSILON = 1e-8  # Python scalar: uses MLX weak-type promotion (no closure capture risk)

# ---------------------------------------------------------------------------
# Fused Metal kernel: combines 5-7 separate MLX ops into a single GPU dispatch.
#
# Per-frame computation (one thread per frame):
#   xbeat  = max(eps, beat - downbeat)
#   no     = 1.0 - (beat + downbeat) * 0.5
#   total  = xbeat + downbeat + no
#   out[i] = [xbeat/total, downbeat/total, no/total]
# ---------------------------------------------------------------------------
_FUSED_METRICAL_SOURCE = """
uint idx = thread_position_in_grid.x;
uint N = beat_shape[0];
if (idx >= N) return;

float b = (float)beat[idx];
float d = (float)downbeat[idx];

float xbeat = max(1e-8f, b - d);
float no_act = 1.0f - (b + d) * 0.5f;
float total = xbeat + d + no_act;
float inv = 1.0f / total;

out[idx * 3]     = (T)(xbeat * inv);
out[idx * 3 + 1] = (T)(d * inv);
out[idx * 3 + 2] = (T)(no_act * inv);
"""

_fused_metrical_kernel = mx.fast.metal_kernel(
  name="fused_metrical_prep",
  input_names=["beat", "downbeat"],
  output_names=["out"],
  source=_FUSED_METRICAL_SOURCE,
)

# Minimum frame count to justify custom kernel launch overhead.
_FUSED_KERNEL_MIN_FRAMES = 256


def _fused_metrical_prep(beat: mx.array, downbeat: mx.array) -> mx.array:
  """Fused metrical activation prep via custom Metal kernel."""
  N = beat.shape[0]
  tg = min(256, N)
  return _fused_metrical_kernel(
    inputs=[beat, downbeat],
    template=[("T", beat.dtype)],
    grid=(N, 1, 1),
    threadgroup=(tg, 1, 1),
    output_shapes=[(N, 3)],
    output_dtypes=[beat.dtype],
  )[0]


def _mlx_metrical_prep(beat: mx.array, downbeat: mx.array) -> mx.array:
  """Reference MLX implementation (fallback for small inputs)."""
  xbeat = mx.maximum(_EPSILON, beat - downbeat)
  no = 1.0 - (beat + downbeat) * 0.5
  combined = mx.stack([xbeat, downbeat, no], axis=-1)
  norm = mx.sum(combined, axis=-1, keepdims=True)
  return combined / norm


def postprocess_metrical_structure_mlx(
  logits: AllInOneOutput,
  cfg: Config,
  prob_beat: np.ndarray = None,
  prob_downbeat: np.ndarray = None,
  prob_beat_mx: mx.array = None,
  prob_downbeat_mx: mx.array = None,
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

  # Use pre-computed MLX arrays directly to avoid NumPy->MLX round-trip
  if prob_beat_mx is not None and prob_downbeat_mx is not None:
    activations_beat = prob_beat_mx
    activations_downbeat = prob_downbeat_mx
  elif prob_beat is not None and prob_downbeat is not None:
    activations_beat = mx.array(prob_beat)
    activations_downbeat = mx.array(prob_downbeat)
  else:
    # Fused sigmoid operations
    activations_beat = mx.sigmoid(logits.logits_beat[0])
    activations_downbeat = mx.sigmoid(logits.logits_downbeat[0])

  # Fuse activation combination + normalization into a single GPU dispatch.
  N = activations_beat.shape[0]
  if N >= _FUSED_KERNEL_MIN_FRAMES:
    activations_combined = _fused_metrical_prep(activations_beat, activations_downbeat)
  else:
    activations_combined = _mlx_metrical_prep(activations_beat, activations_downbeat)

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
