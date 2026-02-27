import importlib
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from allin1_mlx.postprocessing.functional_mlx import postprocess_functional_structure_mlx
from allin1_mlx.sonify import _sonify_metronome
from allin1_mlx.typings import AllInOneOutput, AnalysisResult


def test_postprocess_functional_structure_handles_no_boundaries():
  cfg = SimpleNamespace(
    min_hops_per_beat=24,
    fps=100,
    hop_size=441,
    sample_rate=44100,
  )
  logits = AllInOneOutput(
    logits_section=mx.full((1, 100), -100.0),
    logits_function=mx.zeros((1, 10, 100)),
  )

  segments = postprocess_functional_structure_mlx(logits, cfg)
  assert len(segments) >= 1
  assert segments[0].start == 0.0
  assert segments[-1].end > 0


def test_sonify_metronome_handles_empty_downbeats():
  result = AnalysisResult(
    path=Path('/tmp/test.wav'),
    bpm=120,
    beats=[1.0, 2.0, 3.0],
    downbeats=[],
    beat_positions=[1, 2, 3],
    segments=[],
  )

  y = _sonify_metronome(result, length=44100)
  assert y.shape == (44100,)


def test_visualize_plot_handles_empty_segments(monkeypatch):
  viz = importlib.import_module('allin1_mlx.visualize')

  class _DummySeparator:
    def __init__(self, model, progress):
      self.model = object()
      self.samplerate = 44100

  monkeypatch.setattr(viz, 'Separator', _DummySeparator)
  monkeypatch.setattr(viz, '_load_audio', lambda path, model: np.zeros((2, 44100), dtype=np.float32))

  result = AnalysisResult(
    path=Path('/tmp/test.wav'),
    bpm=120,
    beats=[1.0, 2.0],
    downbeats=[1.0],
    beat_positions=[1, 2],
    segments=[],
  )

  fig = viz._plot(result)
  assert fig is not None
