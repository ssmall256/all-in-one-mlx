import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional
from urllib.error import URLError
from urllib.request import urlopen

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

_FALLBACK_REVISION = 'da5f3474503fde41860b454a48bc9e7899cd5dfa'
_FALLBACK_BASE_URL = (
  f'https://raw.githubusercontent.com/ssmall256/all-in-one-mlx/{_FALLBACK_REVISION}/mlx-weights'
)
_FALLBACK_CACHE_DIR = Path.home() / '.cache' / 'all-in-one-mlx' / 'weights'
_FALLBACK_DOWNLOAD_TIMEOUT = 60
_FALLBACK_FILES = {
  'harmonix-fold0': (
    ('harmonix-fold0_mlx.npz', '2d04d162842b048c34d9476c061f2612e4b9cd339d264aa2fb6ec88c59fde1c0'),
    ('harmonix-fold0_mlx.yaml', '669edb57eb2e1a527081b0e1c41455314a5a1b093f30b2407d9ba356e6f0a2d5'),
  ),
  'harmonix-fold1': (
    ('harmonix-fold1_mlx.npz', '1b83331d32a80762e92947c74be32dbc7d10cbed1abf980deab63bb3de556c8b'),
    ('harmonix-fold1_mlx.yaml', '74cdc5aeac5f22d53249339c3cdb272d10ea4b6d758187d1eeaae090ed291f25'),
  ),
  'harmonix-fold2': (
    ('harmonix-fold2_mlx.npz', 'de816995b2132c8fc20a3547829d3fe5de342a6c58fd38ce8c93f2c85bc6d5e8'),
    ('harmonix-fold2_mlx.yaml', 'df7f033210b6f96e5de32ed1e759cc864adb3247bc4f74cd271f48dacd9e3b37'),
  ),
  'harmonix-fold3': (
    ('harmonix-fold3_mlx.npz', 'e9aaa20b6fbbff8416b50d2a9496c3ecc24b2752c5bc2765876a3a5ec9725e9c'),
    ('harmonix-fold3_mlx.yaml', '81ffa99ef2e37a6f87cef0204a03b1dd8409cf9985a76af83616636642830688'),
  ),
  'harmonix-fold4': (
    ('harmonix-fold4_mlx.npz', '1aa4bcfdf29908121107e4655be7f5baae9bcd49afc5c9763441033df8a20513'),
    ('harmonix-fold4_mlx.yaml', 'bcb7e5c82bbfa126f9302d601d40cae4853ea8dd92ea10f0a2f53f446a30a494'),
  ),
  'harmonix-fold5': (
    ('harmonix-fold5_mlx.npz', '435bebb4ebe55b158d2656e541e737b66fb844725dd56f5a2ff0dc2964046005'),
    ('harmonix-fold5_mlx.yaml', '7e3e3cfee4e4d8a643dfd325e5c126db1f8a0cb925ebe4d2a2fb936599967e50'),
  ),
  'harmonix-fold6': (
    ('harmonix-fold6_mlx.npz', '0f86b5f00468f1546f558fdc72a4cb3fa91342c039bbf381d6c85f0e31d0b76d'),
    ('harmonix-fold6_mlx.yaml', 'd24fba03d24637d7c1026ac40f283a60617af0effccc7869643d5d53d401e93c'),
  ),
  'harmonix-fold7': (
    ('harmonix-fold7_mlx.npz', '682a87d253dbdcd1d9e343c6fae73e118d32441d86a3a409714779fc49ad327f'),
    ('harmonix-fold7_mlx.yaml', 'eba56bdf405973a6d762b85d66743a73da1ba5e268311c33be3ecb1f78fb51b9'),
  ),
}


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as file:
    for chunk in iter(lambda: file.read(1024 * 1024), b''):
      digest.update(chunk)
  return digest.hexdigest()


def _download_verified_file(filename: str, expected_sha256: str, cache_dir: Path) -> Path:
  destination = cache_dir / filename
  if destination.is_file() and _sha256(destination) == expected_sha256:
    return destination

  if destination.exists():
    destination.unlink()
  cache_dir.mkdir(parents=True, exist_ok=True)

  file_descriptor, temporary_name = tempfile.mkstemp(prefix=f'.{filename}.', suffix='.tmp', dir=cache_dir)
  os.close(file_descriptor)
  temporary_path = Path(temporary_name)
  url = f'{_FALLBACK_BASE_URL}/{filename}'
  try:
    try:
      with urlopen(url, timeout=_FALLBACK_DOWNLOAD_TIMEOUT) as response, temporary_path.open('wb') as output:
        shutil.copyfileobj(response, output)
    except (OSError, URLError) as exc:
      raise RuntimeError(
        f'Could not download model file {filename} from {url}. '
        'Check the network connection and try again.'
      ) from exc

    actual_sha256 = _sha256(temporary_path)
    if actual_sha256 != expected_sha256:
      raise RuntimeError(
        f'Checksum verification failed for {filename}: '
        f'expected {expected_sha256}, received {actual_sha256}. '
        'The downloaded file was discarded.'
      )
    os.replace(temporary_path, destination)
    return destination
  finally:
    temporary_path.unlink(missing_ok=True)


def _resolve_fallback_model(model_name: str, cache_dir: Optional[Path] = None) -> Path:
  files = _FALLBACK_FILES.get(model_name)
  if files is None:
    raise ValueError(f'No verified fallback weights are available for model {model_name}.')

  resolved_cache_dir = _FALLBACK_CACHE_DIR if cache_dir is None else Path(cache_dir)
  resolved = [
    _download_verified_file(filename, expected_sha256, resolved_cache_dir)
    for filename, expected_sha256 in files
  ]
  return resolved[0]


def _resolve_optional_provider(model_name: str) -> Optional[Path]:
  try:
    from mlx_weights import resolve_converted_model

    result = resolve_converted_model(f'allin1/{model_name}', convert_if_missing=True)
    if result is None:
      return None
    weights_path = Path(result.primary)
    if not weights_path.is_file():
      return None
    _resolve_config_path(weights_path, None)
    return weights_path
  except Exception:
    return None


def _resolve_weights_path(
  model_name: Optional[str],
  weights_path: Optional[PathLike],
  weights_dir: Optional[PathLike],
) -> Path:
  if weights_path is not None:
    explicit_path = Path(weights_path)
    if not explicit_path.is_file():
      raise FileNotFoundError(f'Could not find MLX weights at {explicit_path}.')
    return explicit_path
  if model_name is None:
    raise ValueError('model_name is required when weights_path is not provided.')
  if weights_dir is not None:
    base_dir = Path(weights_dir)
    for suffix in ('.safetensors', '.npz'):
      candidate = base_dir / f'{model_name}_mlx{suffix}'
      if candidate.is_file():
        return candidate
    raise FileNotFoundError(
      f'Could not find MLX weights for {model_name}. '
      f'Expected {model_name}_mlx.safetensors or {model_name}_mlx.npz in {base_dir}.'
    )

  provider_path = _resolve_optional_provider(model_name)
  if provider_path is not None:
    return provider_path
  return _resolve_fallback_model(model_name)


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
