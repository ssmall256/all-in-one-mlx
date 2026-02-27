from pathlib import Path
from tempfile import TemporaryDirectory

import allin1_mlx

CWD = Path(__file__).resolve().parent


def _get_result():
  return allin1_mlx.analyze(CWD / 'test.mp3')


def test_sonify():
  result = _get_result()
  y, sr = allin1_mlx.sonify(result)
  assert y.shape[0] == 2  # stereo
  assert sr == 44100


def test_sonify_save():
  result = _get_result()
  with TemporaryDirectory() as tmpdir:
    allin1_mlx.sonify(result, out_dir=tmpdir)
    out_path = Path(tmpdir) / 'test.sonif.wav'
    assert out_path.is_file()
