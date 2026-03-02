import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from allin1_mlx.demix import demix


def test_demix_uses_default_shifts(monkeypatch, tmp_path):
  captured = {"shifts": None}

  class DummySeparator:
    samplerate = 44100

    def __init__(self, model, progress, shifts=1):
      captured["shifts"] = shifts

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
  demix([audio_path], tmp_path / "demix", overwrite=True)

  # all-in-one-mlx should use demucs-mlx defaults (shifts=1) unless explicitly configured.
  assert captured["shifts"] == 1
