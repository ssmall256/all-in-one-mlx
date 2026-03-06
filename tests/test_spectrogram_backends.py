import importlib
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from scipy.io import wavfile

from allin1_mlx.spectrogram import (
  extract_spectrograms,
  get_spec_backend_guard_state,
  reset_spec_backend_guard_state,
  spectrogram_from_stems,
)


def _write_stem(path: Path, sr: int, freq: float) -> None:
  t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
  signal = 0.2 * np.sin(2 * np.pi * freq * t)
  data = (signal * np.iinfo(np.int16).max).astype(np.int16)
  wavfile.write(str(path), sr, data)


def _make_demix_dir(root: Path) -> Path:
  track_dir = root / "htdemucs" / "track1"
  track_dir.mkdir(parents=True, exist_ok=True)
  sr = 44100
  _write_stem(track_dir / "bass.wav", sr, 110.0)
  _write_stem(track_dir / "drums.wav", sr, 220.0)
  _write_stem(track_dir / "other.wav", sr, 330.0)
  _write_stem(track_dir / "vocals.wav", sr, 440.0)
  return track_dir


def _load_spec(path: Path) -> np.ndarray:
  return np.load(str(path))


def _madmom_available():
  try:
    import madmom  # noqa: F401
    return True
  except Exception:
    return False


def test_spectrogram_mlx_produces_output():
  with TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    demix_dir = tmp_path / "demix"
    track_dir = _make_demix_dir(demix_dir)

    mlx_dir = tmp_path / "spec_mlx"
    mlx_paths = extract_spectrograms(
      [track_dir],
      mlx_dir,
      multiprocess=False,
      backend="mlx",
    )

    assert len(mlx_paths) == 1
    spec = _load_spec(mlx_paths[0])
    assert spec.ndim >= 2
    assert spec.shape[-1] == 81  # 12 bands per octave filterbank


def test_spectrogram_mlx_fast_mlx_guard_thresholds():
  with TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    demix_dir = tmp_path / "demix"
    track_dir = _make_demix_dir(demix_dir)

    mlx_dir = tmp_path / "spec_mlx"
    fast_dir = tmp_path / "spec_fast"
    mlx_paths = extract_spectrograms(
      [track_dir],
      mlx_dir,
      multiprocess=False,
      backend="mlx",
    )
    reset_spec_backend_guard_state("mlx_fast")
    fast_paths = extract_spectrograms(
      [track_dir],
      fast_dir,
      multiprocess=False,
      backend="mlx_fast",
      spec_fast_guard=False,
    )

    spec_mlx = _load_spec(mlx_paths[0])
    spec_fast = _load_spec(fast_paths[0])
    assert spec_mlx.shape == spec_fast.shape
    diff = np.abs(spec_mlx - spec_fast)
    assert float(diff.max()) <= 1e-3
    assert float(diff.mean()) <= 1e-4


@pytest.mark.skipif(not _madmom_available(), reason="madmom not installed")
def test_spectrogram_madmom_mlx_parity():
  with TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    demix_dir = tmp_path / "demix"
    track_dir = _make_demix_dir(demix_dir)

    madmom_dir = tmp_path / "spec_madmom"
    mlx_dir = tmp_path / "spec_mlx"

    madmom_paths = extract_spectrograms(
      [track_dir],
      madmom_dir,
      multiprocess=False,
      backend="madmom",
    )
    mlx_paths = extract_spectrograms(
      [track_dir],
      mlx_dir,
      multiprocess=False,
      backend="mlx",
    )

    spec_madmom = _load_spec(madmom_paths[0])
    spec_mlx = _load_spec(mlx_paths[0])
    assert spec_madmom.shape == spec_mlx.shape
    diff = np.abs(spec_madmom - spec_mlx)
    assert float(diff.max()) <= 1e-4


def test_spectrogram_guard_forced_fallback_mocked(monkeypatch):
  spectrogram_module = importlib.import_module("allin1_mlx.spectrogram")
  stems = {
    "bass": np.zeros(64, dtype=np.float32),
    "drums": np.zeros(64, dtype=np.float32),
    "other": np.zeros(64, dtype=np.float32),
    "vocals": np.zeros(64, dtype=np.float32),
  }

  def fake_fast(signals, sample_rate, return_mx=False):
    return np.ones((4, 2, 3), dtype=np.float32)

  def fake_ref(signals, sample_rate, return_mx=False):
    return np.zeros((4, 2, 3), dtype=np.float32)

  monkeypatch.setattr(spectrogram_module, "_mlx_log_spectrogram_fast_batch", fake_fast)
  monkeypatch.setattr(spectrogram_module, "_mlx_log_spectrogram_batch", fake_ref)

  reset_spec_backend_guard_state("mlx_fast")
  spec = spectrogram_from_stems(
    stems,
    sample_rate=44100,
    backend="mlx_fast",
    spec_fast_guard=True,
    spec_fast_guard_max_abs=1e-6,
    spec_fast_guard_mean_abs=1e-6,
  )
  state = get_spec_backend_guard_state()
  assert np.all(spec == 0.0)
  assert state["requested_backend"] == "mlx_fast"
  assert state["effective_backend"] == "mlx"
  assert state["guard_triggered"] is True
