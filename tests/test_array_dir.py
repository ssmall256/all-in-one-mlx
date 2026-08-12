from pathlib import Path

import numpy as np

from allin1_mlx.analyze import analyze
from allin1_mlx.cli import make_parser
from allin1_mlx.helpers import save_results
from allin1_mlx.typings import AnalysisResult, Segment
from allin1_mlx.utils import load_result


def _result(*, path: Path = Path('/fake/path/test.wav'), include_arrays: bool = True) -> AnalysisResult:
  return AnalysisResult(
    path=path,
    bpm=120,
    beats=[0.37, 1.23],
    downbeats=[0.37],
    beat_positions=[1, 2],
    segments=[Segment(start=0.0, end=2.0, label='intro')],
    activations={'beat': np.zeros(4, dtype=np.float32)} if include_arrays else None,
    embeddings=np.zeros((2, 3), dtype=np.float32) if include_arrays else None,
  )


def test_arrays_stay_beside_json_by_default(tmp_path):
  out_dir = tmp_path / 'struct'

  save_results(_result(), out_dir)

  assert (out_dir / 'test.json').is_file()
  assert (out_dir / 'test.activ.npz').is_file()
  assert (out_dir / 'test.embed.npy').is_file()


def test_array_dir_separates_and_reloads_arrays(tmp_path):
  out_dir = tmp_path / 'struct'
  array_dir = tmp_path / 'nested' / 'arrays'

  save_results(_result(), out_dir, array_dir=array_dir)

  assert (out_dir / 'test.json').is_file()
  assert not (out_dir / 'test.activ.npz').exists()
  assert not (out_dir / 'test.embed.npy').exists()
  assert (array_dir / 'test.activ.npz').is_file()
  assert (array_dir / 'test.embed.npy').is_file()

  loaded = load_result(out_dir / 'test.json', array_dir=array_dir)
  assert loaded.activations is not None
  assert np.array_equal(loaded.activations['beat'], np.zeros(4, dtype=np.float32))
  assert loaded.embeddings is not None
  assert np.array_equal(loaded.embeddings, np.zeros((2, 3), dtype=np.float32))


def test_unused_array_dir_is_not_created(tmp_path):
  array_dir = tmp_path / 'arrays'

  save_results(_result(include_arrays=False), tmp_path / 'struct', array_dir=array_dir)

  assert not array_dir.exists()


def test_analyze_reloads_cached_arrays_from_array_dir(tmp_path):
  audio_path = tmp_path / 'song.wav'
  audio_path.write_bytes(b'RIFF')
  out_dir = tmp_path / 'struct'
  array_dir = tmp_path / 'arrays'
  save_results(_result(path=audio_path), out_dir, array_dir=array_dir)

  loaded = analyze(
    audio_path,
    out_dir=out_dir,
    include_activations=True,
    include_embeddings=True,
    array_dir=array_dir,
  )

  assert loaded.activations is not None
  assert loaded.embeddings is not None


def test_cli_accepts_array_dir(tmp_path):
  array_dir = tmp_path / 'arrays'

  args = make_parser().parse_args(['song.wav', '--array-dir', str(array_dir)])

  assert args.array_dir == array_dir
