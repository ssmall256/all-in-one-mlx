import matplotlib

matplotlib.use('Agg')

from pathlib import Path
from tempfile import TemporaryDirectory

import allin1_mlx

CWD = Path(__file__).resolve().parent


def test_visualize():
  result = allin1_mlx.analyze(
    paths=CWD / 'test.mp3',
    keep_byproducts=True,
  )
  fig = allin1_mlx.visualize(result)
  assert fig is not None


def test_visualize_save():
  result = allin1_mlx.analyze(
    paths=CWD / 'test.mp3',
    keep_byproducts=True,
  )
  with TemporaryDirectory() as tmpdir:
    allin1_mlx.visualize(result, out_dir=tmpdir)
    assert (Path(tmpdir) / 'test.pdf').is_file()
