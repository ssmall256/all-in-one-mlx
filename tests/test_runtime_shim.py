import importlib
import os
import sys


def test_import_is_lazy_and_exposes_runtime_hooks():
  sys.modules.pop("allin1_mlx", None)
  sys.modules.pop("allin1_mlx.analyze", None)
  sys.modules.pop("allin1_mlx.config", None)

  mod = importlib.import_module("allin1_mlx")
  assert callable(mod.configure)
  assert callable(mod.setup)
  assert callable(mod.init)
  assert callable(mod.enable_optimizations)
  assert callable(mod.apply_runtime_patches)
  assert "allin1_mlx.analyze" not in sys.modules
  assert "allin1_mlx.config" not in sys.modules


def test_runtime_hooks_apply_defaults_and_respect_existing_values():
  import allin1_mlx

  original = dict(os.environ)
  try:
    os.environ.pop("NATTEN_MLX", None)
    os.environ.pop("NATTEN_MLX_BACKEND", None)
    os.environ.pop("NATTEN_MLX_COMPILE", None)
    applied = allin1_mlx.configure()
    assert applied["NATTEN_MLX"] == "1"
    assert applied["NATTEN_MLX_BACKEND"] == "metal"
    assert applied["NATTEN_MLX_COMPILE"] == "1"

    os.environ["NATTEN_MLX"] = "0"
    applied = allin1_mlx.enable_optimizations()
    assert "NATTEN_MLX" not in applied
    assert os.environ["NATTEN_MLX"] == "0"

    applied = allin1_mlx.apply_runtime_patches(force=True)
    assert applied["NATTEN_MLX"] == "1"
    assert os.environ["NATTEN_MLX"] == "1"
  finally:
    os.environ.clear()
    os.environ.update(original)


def test_module_alias_works():
  alias_mod = importlib.import_module("all_in_one_mlx")
  assert callable(alias_mod.configure)
