import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from allin1_mlx.helpers import run_inference_mlx
from allin1_mlx.typings import AllInOneOutput


def test_run_inference_mlx_smoke():
  try:
    import mlx.core as mx
  except Exception:
    return

  frames = 512
  num_labels = 10
  num_instruments = 4
  dim_embed = 24

  cfg = SimpleNamespace(
    sample_rate=44100,
    hop_size=441,
    fps=100,
    best_threshold_downbeat=0.19,
    min_hops_per_beat=24,
    bpm_min=55,
    bpm_max=240,
    data=SimpleNamespace(
      num_labels=num_labels,
      num_instruments=num_instruments,
    ),
  )

  def _logits_beat(n):
    """Generate plausible beat logits with peaks every ~50 frames."""
    logits = mx.full((n,), -3.0)
    for i in range(25, n, 50):
      logits[i] = 3.0
    return logits

  def _logits_section(n):
    """Generate plausible section boundary logits."""
    logits = mx.full((n,), -3.0)
    for i in [0, 200, 400]:
      if i < n:
        logits[i] = 3.0
    return logits

  def _logits_function(nl, n):
    """Generate plausible function logits (verse/chorus pattern)."""
    logits = mx.zeros((nl, n))
    half = n // 2
    logits = logits.at[8, :half].add(5.0)
    logits = logits.at[9, half:].add(5.0)
    return logits

  class DummyModel:
    def __init__(self, cfg):
      self.cfg = cfg

    def __call__(self, spec, return_embeddings=False):
      beat = _logits_beat(frames)[None, :]
      downbeat = _logits_beat(frames)[None, :]
      section = _logits_section(frames)[None, :]
      function = _logits_function(num_labels, frames)[None, :, :]
      embeddings = mx.zeros((1, num_instruments, frames, dim_embed)) if return_embeddings else None
      return AllInOneOutput(
        logits_beat=beat,
        logits_downbeat=downbeat,
        logits_section=section,
        logits_function=function,
        embeddings=embeddings,
      )

  model = DummyModel(cfg)

  with tempfile.TemporaryDirectory() as tmpdir:
    spec_path = Path(tmpdir) / "spec.npy"
    np.save(spec_path, np.random.randn(num_instruments, frames, 81).astype(np.float32))

    result = run_inference_mlx(
      path=Path("dummy.wav"),
      spec_path=spec_path,
      model=model,
      include_activations=True,
      include_embeddings=True,
    )

  assert isinstance(result.beats, list)
  assert isinstance(result.downbeats, list)
  assert isinstance(result.segments, list)
  assert len(result.beats) > 0
  assert len(result.segments) > 0
  assert result.embeddings is not None
