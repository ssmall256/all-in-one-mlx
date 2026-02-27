from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf

from ..typings import PathLike
from .allinone_mlx import AllInOneMLX
from .ensemble_mlx import EnsembleMLX

NAME_TO_FILE = {
  'harmonix-fold0': 'harmonix-fold0-0vra4ys2.pth',
  'harmonix-fold1': 'harmonix-fold1-3ozjhtsj.pth',
  'harmonix-fold2': 'harmonix-fold2-gmgo0nsy.pth',
  'harmonix-fold3': 'harmonix-fold3-i92b7m8p.pth',
  'harmonix-fold4': 'harmonix-fold4-1bql5qo0.pth',
  'harmonix-fold5': 'harmonix-fold5-x4z5zeef.pth',
  'harmonix-fold6': 'harmonix-fold6-x7t226rq.pth',
  'harmonix-fold7': 'harmonix-fold7-qwwskhg6.pth',
}

ENSEMBLE_MODELS = {
  'harmonix-all': [
    'harmonix-fold0',
    'harmonix-fold1',
    'harmonix-fold2',
    'harmonix-fold3',
    'harmonix-fold4',
    'harmonix-fold5',
    'harmonix-fold6',
    'harmonix-fold7',
  ],
}


def _resolve_weights_path(
  model_name: Optional[str],
  weights_path: Optional[PathLike],
  weights_dir: Optional[PathLike],
) -> Path:
  if weights_path is not None:
    return Path(weights_path)
  if model_name is None:
    raise ValueError("model_name is required when weights_path is not provided.")
  base_dir = Path(weights_dir) if weights_dir is not None else Path("mlx-weights")
  for suffix in (".safetensors", ".npz"):
    candidate = base_dir / f"{model_name}_mlx{suffix}"
    if candidate.is_file():
      return candidate
  raise FileNotFoundError(
    f"Could not find MLX weights for {model_name}. "
    f"Expected {model_name}_mlx.safetensors or {model_name}_mlx.npz in {base_dir}."
  )


def _resolve_config_path(weights_path: Path, config_path: Optional[PathLike]) -> Path:
  if config_path is not None:
    return Path(config_path)
  for suffix in (".yaml", ".yml", ".json"):
    candidate = weights_path.with_suffix(suffix)
    if candidate.is_file():
      return candidate
  raise FileNotFoundError(
    f"Could not find config file next to {weights_path}. "
    "Expected a .yaml/.yml/.json config."
  )


def load_pretrained_model_mlx(
  model_name: Optional[str] = None,
  weights_path: Optional[PathLike] = None,
  weights_dir: Optional[PathLike] = None,
  config_path: Optional[PathLike] = None,
  strict: bool = True,
  ensemble_parallel: bool = True,
):
  if model_name in ENSEMBLE_MODELS:
    return load_ensemble_model_mlx(model_name, weights_dir, strict, ensemble_parallel)

  weights_path = _resolve_weights_path(model_name, weights_path, weights_dir)
  config_path = _resolve_config_path(weights_path, config_path)
  config = OmegaConf.load(config_path)

  model = AllInOneMLX(config)
  model.load_weights(str(weights_path), strict=strict)
  model.eval()

  return model


def load_ensemble_model_mlx(
  model_name: Optional[str] = None,
  weights_dir: Optional[PathLike] = None,
  strict: bool = True,
  parallel: bool = True,
):
  if model_name not in ENSEMBLE_MODELS:
    raise ValueError(f"Unknown ensemble name: {model_name}")

  models = []
  for submodel_name in ENSEMBLE_MODELS[model_name]:
    model = load_pretrained_model_mlx(
      model_name=submodel_name,
      weights_dir=weights_dir,
      strict=strict,
      ensemble_parallel=parallel,
    )
    models.append(model)

  ensemble = EnsembleMLX(models, parallel=parallel)
  ensemble.eval()
  return ensemble
