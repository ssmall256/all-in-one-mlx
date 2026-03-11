import importlib
import sys
import types
from types import SimpleNamespace


def test_default_loader_uses_mlx_weights_cache_and_conversion(tmp_path, monkeypatch):
  fake_allinone = types.ModuleType("allin1_mlx.models.allinone_mlx")
  fake_ensemble = types.ModuleType("allin1_mlx.models.ensemble_mlx")

  class DummyModel:
    def __init__(self, config):
      self.config = config
      self.loaded = None
      self.strict = None
      self.evaluated = False

    def load_weights(self, path, strict=True):
      self.loaded = path
      self.strict = strict

    def eval(self):
      self.evaluated = True

  class DummyEnsemble:
    def __init__(self, models, parallel=True):
      self.models = models
      self.parallel = parallel
      self.evaluated = False

    def eval(self):
      self.evaluated = True

  fake_allinone.AllInOneMLX = DummyModel
  fake_ensemble.EnsembleMLX = DummyEnsemble
  monkeypatch.setitem(sys.modules, "allin1_mlx.models.allinone_mlx", fake_allinone)
  monkeypatch.setitem(sys.modules, "allin1_mlx.models.ensemble_mlx", fake_ensemble)

  sys.modules.pop("allin1_mlx.models.loaders_mlx", None)
  loaders_mlx = importlib.import_module("allin1_mlx.models.loaders_mlx")

  monkeypatch.setenv("MLX_WEIGHTS_CACHE", str(tmp_path))
  weights_path = tmp_path / "all-in-one-mlx" / "harmonix-fold0_mlx.safetensors"
  config_path = tmp_path / "all-in-one-mlx" / "harmonix-fold0_mlx.json"
  seen = {}

  def fake_convert_model(name, input_path=None, output_path=None, extra_output_paths=None):
    seen["name"] = name
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path.write_bytes(b"weights")
    config_path.write_text("{}", encoding="utf-8")
    return SimpleNamespace(primary=weights_path)

  monkeypatch.setattr(loaders_mlx, "convert_model", fake_convert_model)
  monkeypatch.setattr(loaders_mlx.OmegaConf, "load", lambda path: {"config_path": str(path)})

  model = loaders_mlx.load_pretrained_model_mlx(model_name="harmonix-fold0")
  assert seen["name"] == "allin1/harmonix-fold0"
  assert model.loaded == str(weights_path)
  assert model.strict is True
  assert model.evaluated is True
  assert model.config == {"config_path": str(config_path)}
