from pathlib import Path

import allin1_mlx

CWD = Path(__file__).resolve().parent


def test_analyze():
  allin1_mlx.analyze(CWD / 'test.mp3')
