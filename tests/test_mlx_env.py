import os

from allin1_mlx.analyze import _ensure_mlx_env


def test_ensure_mlx_env_defaults():
  original = dict(os.environ)
  try:
    os.environ.pop("NATTEN_MLX", None)
    os.environ.pop("NATTEN_MLX_BACKEND", None)
    os.environ.pop("NATTEN_MLX_COMPILE", None)
    _ensure_mlx_env("mlx")
    assert os.environ["NATTEN_MLX"] == "1"
    assert os.environ["NATTEN_MLX_BACKEND"] == "metal"
    assert os.environ["NATTEN_MLX_COMPILE"] == "1"
  finally:
    os.environ.clear()
    os.environ.update(original)


def test_ensure_mlx_env_respects_overrides():
  original = dict(os.environ)
  try:
    os.environ["NATTEN_MLX"] = "0"
    os.environ["NATTEN_MLX_BACKEND"] = "mlx"
    os.environ["NATTEN_MLX_COMPILE"] = "0"
    _ensure_mlx_env("mlx")
    assert os.environ["NATTEN_MLX"] == "0"
    assert os.environ["NATTEN_MLX_BACKEND"] == "mlx"
    assert os.environ["NATTEN_MLX_COMPILE"] == "0"
  finally:
    os.environ.clear()
    os.environ.update(original)


def test_ensure_mlx_env_noop_for_non_mlx():
  original = dict(os.environ)
  try:
    os.environ.pop("NATTEN_MLX", None)
    _ensure_mlx_env("cpu")
    assert "NATTEN_MLX" not in os.environ
  finally:
    os.environ.clear()
    os.environ.update(original)
