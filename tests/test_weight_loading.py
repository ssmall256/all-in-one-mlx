import hashlib
import importlib
import io
import sys
import types
from types import SimpleNamespace

import pytest


@pytest.fixture
def loaders_mlx(monkeypatch):
  fake_allinone = types.ModuleType('allin1_mlx.models.allinone_mlx')
  fake_ensemble = types.ModuleType('allin1_mlx.models.ensemble_mlx')

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
  monkeypatch.setitem(sys.modules, 'allin1_mlx.models.allinone_mlx', fake_allinone)
  monkeypatch.setitem(sys.modules, 'allin1_mlx.models.ensemble_mlx', fake_ensemble)

  sys.modules.pop('allin1_mlx.models.loaders_mlx', None)
  module = importlib.import_module('allin1_mlx.models.loaders_mlx')
  monkeypatch.setattr(module.OmegaConf, 'load', lambda path: {'config_path': str(path)})
  yield module
  sys.modules.pop('allin1_mlx.models.loaders_mlx', None)


def _sha256(data):
  return hashlib.sha256(data).hexdigest()


def _install_provider(monkeypatch, resolver):
  provider = types.ModuleType('mlx_weights')
  provider.resolve_converted_model = resolver
  monkeypatch.setitem(sys.modules, 'mlx_weights', provider)


def _configure_fake_downloads(loaders_mlx, monkeypatch, tmp_path, weights=b'weights', config=b'config: true\n'):
  monkeypatch.setattr(loaders_mlx, '_FALLBACK_CACHE_DIR', tmp_path)
  monkeypatch.setattr(
    loaders_mlx,
    '_FALLBACK_FILES',
    {'test-model': (('test-model_mlx.npz', _sha256(weights)), ('test-model_mlx.yaml', _sha256(config)))},
  )
  payloads = {
    'test-model_mlx.npz': weights,
    'test-model_mlx.yaml': config,
  }
  calls = []

  def fake_urlopen(url, timeout):
    calls.append((url, timeout))
    return io.BytesIO(payloads[url.rsplit('/', 1)[-1]])

  monkeypatch.setattr(loaders_mlx, 'urlopen', fake_urlopen)
  return calls


def test_optional_provider_is_preferred(loaders_mlx, tmp_path, monkeypatch):
  weights_path = tmp_path / 'provider' / 'harmonix-fold0_mlx.safetensors'
  config_path = weights_path.with_suffix('.json')
  weights_path.parent.mkdir()
  weights_path.write_bytes(b'weights')
  config_path.write_text('{}', encoding='utf-8')
  seen = {}

  def resolve(name, convert_if_missing=False):
    seen['request'] = (name, convert_if_missing)
    return SimpleNamespace(primary=weights_path)

  _install_provider(monkeypatch, resolve)
  monkeypatch.setattr(
    loaders_mlx,
    '_resolve_fallback_model',
    lambda model_name: pytest.fail('fallback should not run when the provider is usable'),
  )

  model = loaders_mlx.load_pretrained_model_mlx(model_name='harmonix-fold0')

  assert seen['request'] == ('allin1/harmonix-fold0', True)
  assert model.loaded == str(weights_path)
  assert model.config == {'config_path': str(config_path)}
  assert model.strict is True
  assert model.evaluated is True


def test_missing_optional_provider_uses_verified_fallback(loaders_mlx, tmp_path, monkeypatch):
  monkeypatch.setitem(sys.modules, 'mlx_weights', None)
  calls = _configure_fake_downloads(loaders_mlx, monkeypatch, tmp_path)

  resolved = loaders_mlx._resolve_weights_path('test-model', None, None)

  assert resolved == tmp_path / 'test-model_mlx.npz'
  assert (tmp_path / 'test-model_mlx.npz').read_bytes() == b'weights'
  assert (tmp_path / 'test-model_mlx.yaml').read_bytes() == b'config: true\n'
  assert len(calls) == 2


@pytest.mark.parametrize('result', [None, SimpleNamespace(primary='/missing/provider-weights.npz')])
def test_unusable_optional_provider_uses_fallback(loaders_mlx, tmp_path, monkeypatch, result):
  _install_provider(monkeypatch, lambda name, convert_if_missing=False: result)
  fallback_path = tmp_path / 'fallback.npz'
  monkeypatch.setattr(loaders_mlx, '_resolve_fallback_model', lambda model_name: fallback_path)

  assert loaders_mlx._resolve_weights_path('harmonix-fold0', None, None) == fallback_path


def test_failing_optional_provider_uses_fallback(loaders_mlx, tmp_path, monkeypatch):
  def fail(name, convert_if_missing=False):
    raise RuntimeError('provider failure')

  _install_provider(monkeypatch, fail)
  fallback_path = tmp_path / 'fallback.npz'
  monkeypatch.setattr(loaders_mlx, '_resolve_fallback_model', lambda model_name: fallback_path)

  assert loaders_mlx._resolve_weights_path('harmonix-fold0', None, None) == fallback_path


def test_explicit_weights_path_takes_precedence(loaders_mlx, tmp_path, monkeypatch):
  weights_path = tmp_path / 'custom.npz'
  weights_path.write_bytes(b'weights')
  monkeypatch.setattr(
    loaders_mlx,
    '_resolve_optional_provider',
    lambda model_name: pytest.fail('provider should not run for an explicit path'),
  )

  assert loaders_mlx._resolve_weights_path('harmonix-fold0', weights_path, tmp_path / 'unused') == weights_path


def test_explicit_weights_dir_takes_precedence(loaders_mlx, tmp_path, monkeypatch):
  weights_path = tmp_path / 'harmonix-fold0_mlx.npz'
  weights_path.write_bytes(b'weights')
  monkeypatch.setattr(
    loaders_mlx,
    '_resolve_optional_provider',
    lambda model_name: pytest.fail('provider should not run for an explicit directory'),
  )

  assert loaders_mlx._resolve_weights_path('harmonix-fold0', None, tmp_path) == weights_path


def test_invalid_explicit_paths_are_errors(loaders_mlx, tmp_path, monkeypatch):
  monkeypatch.setattr(
    loaders_mlx,
    '_resolve_optional_provider',
    lambda model_name: pytest.fail('provider should not run for invalid explicit paths'),
  )

  with pytest.raises(FileNotFoundError, match='Could not find MLX weights at'):
    loaders_mlx._resolve_weights_path('harmonix-fold0', tmp_path / 'missing.npz', None)
  with pytest.raises(FileNotFoundError, match='Expected harmonix-fold0_mlx'):
    loaders_mlx._resolve_weights_path('harmonix-fold0', None, tmp_path / 'missing-dir')


def test_verified_cache_is_reused_without_network(loaders_mlx, tmp_path, monkeypatch):
  weights = b'cached weights'
  config = b'cached config\n'
  monkeypatch.setattr(
    loaders_mlx,
    '_FALLBACK_FILES',
    {'test-model': (('test-model_mlx.npz', _sha256(weights)), ('test-model_mlx.yaml', _sha256(config)))},
  )
  (tmp_path / 'test-model_mlx.npz').write_bytes(weights)
  (tmp_path / 'test-model_mlx.yaml').write_bytes(config)
  monkeypatch.setattr(loaders_mlx, 'urlopen', lambda *args, **kwargs: pytest.fail('network should not run'))

  assert loaders_mlx._resolve_fallback_model('test-model', tmp_path) == tmp_path / 'test-model_mlx.npz'


def test_corrupt_cache_entry_is_downloaded_again(loaders_mlx, tmp_path, monkeypatch):
  calls = _configure_fake_downloads(loaders_mlx, monkeypatch, tmp_path)
  (tmp_path / 'test-model_mlx.npz').write_bytes(b'corrupt')
  (tmp_path / 'test-model_mlx.yaml').write_bytes(b'config: true\n')

  resolved = loaders_mlx._resolve_fallback_model('test-model', tmp_path)

  assert resolved.read_bytes() == b'weights'
  assert len(calls) == 1


def test_checksum_rejection_discards_download(loaders_mlx, tmp_path, monkeypatch):
  monkeypatch.setattr(
    loaders_mlx,
    '_FALLBACK_FILES',
    {'test-model': (('test-model_mlx.npz', _sha256(b'expected')), ('test-model_mlx.yaml', _sha256(b'config')))},
  )
  monkeypatch.setattr(loaders_mlx, 'urlopen', lambda url, timeout: io.BytesIO(b'untrusted'))

  with pytest.raises(RuntimeError, match='Checksum verification failed'):
    loaders_mlx._resolve_fallback_model('test-model', tmp_path)

  assert not (tmp_path / 'test-model_mlx.npz').exists()
  assert list(tmp_path.iterdir()) == []


def test_network_error_is_actionable_and_temporary_file_is_removed(loaders_mlx, tmp_path, monkeypatch):
  monkeypatch.setattr(
    loaders_mlx,
    '_FALLBACK_FILES',
    {'test-model': (('test-model_mlx.npz', _sha256(b'weights')), ('test-model_mlx.yaml', _sha256(b'config')))},
  )

  def fail(url, timeout):
    raise OSError('offline')

  monkeypatch.setattr(loaders_mlx, 'urlopen', fail)

  with pytest.raises(RuntimeError, match='Check the network connection'):
    loaders_mlx._resolve_fallback_model('test-model', tmp_path)

  assert list(tmp_path.iterdir()) == []
