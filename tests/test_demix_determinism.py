import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from allin1_mlx.cli import make_parser
from allin1_mlx.demix import demix
from allin1_mlx.typings import AnalysisResult, Segment

analyze_module = importlib.import_module('allin1_mlx.analyze')


@pytest.mark.parametrize('seed', [None, 1234])
def test_demix_uses_default_shifts_and_forwards_seed(monkeypatch, tmp_path, seed):
  captured = {'seed': object(), 'shifts': None}

  class DummySeparator:
    samplerate = 44100

    def __init__(self, model, progress, shifts=1, seed=None):
      captured['seed'] = seed
      captured['shifts'] = shifts

    def separate_audio_file(self, path):
      stems = {
        "bass": np.zeros((2, 128), dtype=np.float32),
        "drums": np.zeros((2, 128), dtype=np.float32),
        "other": np.zeros((2, 128), dtype=np.float32),
        "vocals": np.zeros((2, 128), dtype=np.float32),
      }
      return None, stems

  def dummy_save_audio(audio, dst, sr):
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Path(dst).write_bytes(b"")

  monkeypatch.setitem(
    sys.modules,
    "demucs_mlx.api",
    SimpleNamespace(Separator=DummySeparator, save_audio=dummy_save_audio),
  )

  audio_path = tmp_path / "song.wav"
  audio_path.write_bytes(b"RIFF")
  demix([audio_path], tmp_path / "demix", overwrite=True, seed=seed)

  # all-in-one-mlx should use demucs-mlx defaults (shifts=1) unless explicitly configured.
  assert captured['shifts'] == 1
  assert captured['seed'] == seed


@pytest.mark.parametrize('seed', [None, 4321])
def test_in_memory_analyze_forwards_demix_seed(monkeypatch, tmp_path, seed):
  audio_path = tmp_path / 'song.wav'
  audio_path.write_bytes(b'RIFF')
  captured = {}
  result = AnalysisResult(
    path=audio_path,
    bpm=120,
    beats=[0.0, 0.5],
    downbeats=[0.0],
    beat_positions=[1, 2],
    segments=[Segment(start=0.0, end=1.0, label='intro')],
  )

  class DummySeparator:
    samplerate = 44100

    def separate_audio_file(self, path, *, return_mx=False):
      stems = {
        name: np.zeros((2, 16), dtype=np.float32)
        for name in ('bass', 'drums', 'other', 'vocals')
      }
      return None, stems

  def fake_create_separator(*, seed):
    captured['seed'] = seed
    return DummySeparator()

  monkeypatch.setattr(analyze_module, '_create_separator', fake_create_separator)
  monkeypatch.setattr(analyze_module, 'load_pretrained_model_mlx', lambda **kwargs: object())
  monkeypatch.setattr(
    analyze_module,
    'spectrogram_from_stems',
    lambda *args, **kwargs: np.zeros((4, 8), dtype=np.float32),
  )
  monkeypatch.setattr(analyze_module, 'run_inference_mlx_spec', lambda **kwargs: result)

  resolved = analyze_module.analyze(audio_path, demix_seed=seed)

  assert resolved == result
  assert captured['seed'] == seed


def test_cli_accepts_demix_seed():
  args = make_parser().parse_args(['song.wav', '--demix-seed', '9876'])

  assert args.demix_seed == 9876
