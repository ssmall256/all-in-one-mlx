import importlib

import numpy as np

from allin1_mlx.spectrogram import (
  _should_fallback_to_mlx_fast_ref,
  get_spec_backend_guard_state,
  reset_spec_backend_guard_state,
  spectrogram_from_stems,
)


def _test_stems():
  return {
    "bass": np.zeros(64, dtype=np.float32),
    "drums": np.zeros(64, dtype=np.float32),
    "other": np.zeros(64, dtype=np.float32),
    "vocals": np.zeros(64, dtype=np.float32),
  }


def test_should_fallback_to_mlx_fast_ref_threshold_logic():
  assert _should_fallback_to_mlx_fast_ref(1.1e-3, 0.0, 1e-3, 1e-4)
  assert _should_fallback_to_mlx_fast_ref(0.0, 1.1e-4, 1e-3, 1e-4)
  assert not _should_fallback_to_mlx_fast_ref(1e-3, 1e-4, 1e-3, 1e-4)


def test_guard_state_sticky_after_fallback(monkeypatch):
  spectrogram_module = importlib.import_module("allin1_mlx.spectrogram")
  stems = _test_stems()

  def fake_fast(signals, sample_rate, return_mx=False):
    return np.ones((4, 2, 3), dtype=np.float32)

  def fake_ref(signals, sample_rate, return_mx=False):
    return np.zeros((4, 2, 3), dtype=np.float32)

  monkeypatch.setattr(spectrogram_module, "_mlx_log_spectrogram_fast_batch", fake_fast)
  monkeypatch.setattr(spectrogram_module, "_mlx_log_spectrogram_batch", fake_ref)

  reset_spec_backend_guard_state("mlx_fast")
  spectrogram_from_stems(
    stems,
    sample_rate=44100,
    backend="mlx_fast",
    spec_fast_guard=True,
    spec_fast_guard_max_abs=1e-6,
    spec_fast_guard_mean_abs=1e-6,
  )
  state = get_spec_backend_guard_state()
  assert state["effective_backend"] == "mlx"
  assert state["guard_triggered"] is True

  def fail_fast(*args, **kwargs):
    raise AssertionError("mlx_fast should not be called after guard fallback")

  monkeypatch.setattr(spectrogram_module, "_mlx_log_spectrogram_fast_batch", fail_fast)
  spec = spectrogram_from_stems(
    stems,
    sample_rate=44100,
    backend="mlx_fast",
    spec_fast_guard=True,
  )
  assert np.all(spec == 0.0)


def test_guard_state_no_fallback_when_under_threshold(monkeypatch):
  spectrogram_module = importlib.import_module("allin1_mlx.spectrogram")
  stems = _test_stems()

  def fake_fast(signals, sample_rate, return_mx=False):
    return np.zeros((4, 2, 3), dtype=np.float32)

  def fake_ref(signals, sample_rate, return_mx=False):
    return np.zeros((4, 2, 3), dtype=np.float32)

  monkeypatch.setattr(spectrogram_module, "_mlx_log_spectrogram_fast_batch", fake_fast)
  monkeypatch.setattr(spectrogram_module, "_mlx_log_spectrogram_batch", fake_ref)

  reset_spec_backend_guard_state("mlx_fast")
  spectrogram_from_stems(
    stems,
    sample_rate=44100,
    backend="mlx_fast",
    spec_fast_guard=True,
    spec_fast_guard_max_abs=1e-6,
    spec_fast_guard_mean_abs=1e-6,
  )
  state = get_spec_backend_guard_state()
  assert state["requested_backend"] == "mlx_fast"
  assert state["effective_backend"] == "mlx_fast"
  assert state["guard_checked"] is True
  assert state["guard_triggered"] is False
