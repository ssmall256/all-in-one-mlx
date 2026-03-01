"""Compatibility module name for integrations expecting all_in_one_mlx."""

from __future__ import annotations

from typing import Any

import allin1_mlx as _impl

__all__ = getattr(_impl, "__all__", [])


def __getattr__(name: str) -> Any:
  return getattr(_impl, name)


def __dir__() -> list[str]:
  return sorted(set(dir(_impl)))
