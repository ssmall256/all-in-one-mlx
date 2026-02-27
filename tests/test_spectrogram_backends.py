from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from scipy.io import wavfile

from allin1_mlx.spectrogram import extract_spectrograms


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
