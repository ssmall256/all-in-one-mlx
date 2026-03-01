"""Public API for allin1_mlx.

This module intentionally avoids importing heavy dependencies at import time.
Integrators can import runtime hooks (for example ``configure``) without
pulling the full analysis stack.
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Dict

from .__about__ import __version__

_RUNTIME_DEFAULTS = {
  "NATTEN_MLX": "1",
  "NATTEN_MLX_BACKEND": "metal",
  "NATTEN_MLX_COMPILE": "1",
}

_LAZY_ATTRS = {
  "analyze": ("allin1_mlx.analyze", "analyze"),
  "visualize": ("allin1_mlx.visualize", "visualize"),
  "sonify": ("allin1_mlx.sonify", "sonify"),
  "AnalysisResult": ("allin1_mlx.typings", "AnalysisResult"),
  "HARMONIX_LABELS": ("allin1_mlx.config", "HARMONIX_LABELS"),
  "load_result": ("allin1_mlx.utils", "load_result"),
}


def _apply_runtime_env(*, force: bool, env: Dict[str, str] | None) -> Dict[str, str]:
  values = dict(_RUNTIME_DEFAULTS)
  if env:
    values.update({str(key): str(value) for key, value in env.items()})

  applied = {}
  for key, value in values.items():
    if force or key not in os.environ:
      os.environ[key] = value
      applied[key] = value
  return applied


def configure(*, force: bool = False, env: Dict[str, str] | None = None) -> Dict[str, str]:
  """Apply default runtime environment settings for MLX inference."""
  return _apply_runtime_env(force=force, env=env)


def setup(*, force: bool = False, env: Dict[str, str] | None = None) -> Dict[str, str]:
  """Compatibility alias for integration runtimes expecting setup()."""
  return configure(force=force, env=env)


def init(*, force: bool = False, env: Dict[str, str] | None = None) -> Dict[str, str]:
  """Compatibility alias for integration runtimes expecting init()."""
  return configure(force=force, env=env)


def enable_optimizations(*, force: bool = False, env: Dict[str, str] | None = None) -> Dict[str, str]:
  """Compatibility alias for integration runtimes expecting enable_optimizations()."""
  return configure(force=force, env=env)


def apply_runtime_patches(*, force: bool = False, env: Dict[str, str] | None = None) -> Dict[str, str]:
  """Compatibility alias for integration runtimes expecting apply_runtime_patches()."""
  return configure(force=force, env=env)


def __getattr__(name: str) -> Any:
  target = _LAZY_ATTRS.get(name)
  if target is None:
    raise AttributeError(f"module 'allin1_mlx' has no attribute '{name}'")
  module_name, attr_name = target
  module = importlib.import_module(module_name)
  value = getattr(module, attr_name)
  globals()[name] = value
  return value


def __dir__() -> list[str]:
  return sorted(set(globals()).union(_LAZY_ATTRS))


__all__ = [
  "__version__",
  "configure",
  "setup",
  "init",
  "enable_optimizations",
  "apply_runtime_patches",
  "analyze",
  "visualize",
  "sonify",
  "AnalysisResult",
  "HARMONIX_LABELS",
  "load_result",
]
