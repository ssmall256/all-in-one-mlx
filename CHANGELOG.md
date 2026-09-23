# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.13] - 2026-09-23

### Removed

- The `.gitignore` entries added in 1.0.12 named specific editor and assistant
  tooling. They live in a global ignore file instead, so the repository does not
  carry them.

## [1.0.12] - 2026-09-23

### Removed

- Local working notes dropped from the repository. They were already excluded
  from the sdist, so PyPI was never affected.

## [1.0.11] - 2026-09-23

### Fixed

- Removed developer machine paths from local working notes kept in the
  repository (excluded from the sdist, so PyPI was never affected).

## [1.0.10] - 2026-09-23

### Changed

- Dependency floors raised to the current stack: `demucs-mlx>=1.4.11`,
  `mlx-audio-io>=1.3.14`. `uv.lock` refreshed to match.

## [1.0.9] - 2026-09-23

### Fixed

- **Analysis failed on MLX 0.32.x with `RuntimeError: There is no Stream(cpu,
  N) in current thread`.** `analyze()` loads the model on a background thread,
  and MLX binds an unevaluated array to the stream of the thread that built it.
  That thread is gone by the time the main thread evaluates a graph containing
  the weights, so the failure surfaced from the forward pass, far from its
  cause. The weights are now materialised on the loading thread — which is also
  the point of loading in the background, since otherwise the work just moved
  to the first inference call.

### Changed

- `demucs-mlx` floor raised to 1.4.10 and `mlx-audio-io` to 1.3.13. 1.4.9 fixed a
  missing threadgroup barrier in the fused GroupNorm Metal kernels, and 1.4.8
  made an unreadable weights cache regenerate instead of raising -- which this
  package hit directly when upgrading.
- The 1.0.7 and 1.0.8 entries describe the dependency change rather than the
  debugging that led to it.

## [1.0.8] - 2026-09-23

### Fixed

- **`numpy` ceiling from 1.0.7 removed.** Current numba supports NumPy 2.4 and
  2.5, so the cap was unnecessary and blocked them.
- **`numba` floor raised to 0.64.0** instead. numba gates the NumPy it will
  import against and raises at import time on anything newer, and it is a hard
  import on the beat/downbeat path. `numba>=0.60.0` let a resolver pick a numba
  whose NumPy ceiling had already been passed. Constraining numba rather than
  capping NumPy lets numba's own metadata keep the bound current: 0.60 tops out
  at NumPy 2.2, 0.64 at 2.5, 0.67 at 2.6.

## [1.0.7] - 2026-09-23

### Changed

- Dependencies moved to the released MLX audio stack: `mlx>=0.31.2,<0.33`,
  `demucs-mlx>=1.4.8,<1.5`, `mlx-audio-io>=1.3.12,<1.4`. The MLX floor matters —
  0.31.0 and 0.31.1 mis-linearize the dispatch grid in their Metal strided
  scatter-add kernel, which corrupts overlap-add accumulation, and the previous
  floor of `>=0.31.0` allowed both.

### Fixed

- `numpy` capped below 2.4 for numba compatibility. Superseded in 1.0.8 by a
  `numba` floor, which tracks the bound automatically.

### Note on upgrading

demucs-mlx 1.4.6 hardened its weight cache, so any cache written before it is
rejected. 1.4.8 regenerates automatically; that conversion needs the extras, so
run `pip install 'demucs-mlx[convert]'` once if you see a conversion error on
your first demix after upgrading.

## [1.0.6] - 2026-08-12

### Added

- Optional `mlx-weights` integration with verified, on-demand checkpoint downloads when the integration is unavailable.
- External four- and six-stem inputs for bypassing source separation.
- `--array-dir` for storing activation and embedding arrays separately from JSON results.
- `--demix-seed` for reproducible Demucs time-shift selection.

### Changed

- Removed the mandatory `mlx-weights` dependency and added a standalone cache at `~/.cache/all-in-one-mlx/weights`.
- Reduced MLX batch inference materialization overhead.
- Bumped `demucs-mlx` to `>=1.4.4` for seeded shift support.
- Updated GitHub Actions workflows to current Node 24 action generations.

## [1.0.5] - 2026-03-06

### Changed

- Bumped `demucs-mlx` to `>=1.4.3` (direct in-memory resampling, no temp file round-trips).
- Bumped `mlx-audio-io` to `>=1.3.9` (auto-selects best resampling quality when `sr` is specified).

## [1.0.3] - 2026-03-02

### Added

- `mlx_fast` parity guard controls in `analyze()` and CLI:
  - `spec_fast_guard`
  - `spec_fast_guard_max_abs`
  - `spec_fast_guard_mean_abs`
- Guard state helpers in `spectrogram.py` to expose requested/effective backend and guard trigger status.
- Reproducible beat parity harness: `scripts/compare_beat_parity.py`.
- Regression tests for guard threshold logic, sticky fallback behavior, and spectrogram backend guard behavior.

### Changed

- `mlx_fast` now performs a one-time parity check against `mlx` by default and falls back automatically when thresholds are exceeded.
- Timing JSON summary now includes spectrogram backend guard metadata.

## [1.0.2] - 2026-03-02

### Changed

- Updated `demucs-mlx` runtime dependency to `>=1.4.0` for corrected shifted inference behavior.
- Adopted Demucs-default shift behavior in integration tests (no hardcoded `shifts=0` expectation).

## [1.0.1] - 2026-03-01

### Added

- Runtime integration shim (`configure`, `setup`, `init`, `enable_optimizations`, `apply_runtime_patches`) for external
  orchestrators.
- Compatibility module alias `all_in_one_mlx` for integrations expecting underscore-style import names.
- Regression tests for lazy import behavior and runtime shim contract.

### Changed

- Made `allin1_mlx` top-level imports lazy to avoid importing full analysis stack at module import time.
- Included `all_in_one_mlx` alias package in wheel and sdist artifacts.

## [1.0.0] - 2026-02-26

### Added

- MLX-native inference for Apple Silicon (M1/M2/M3/M4).
- `natten-mlx` fused Metal kernels for neighborhood attention (1D and 2D).
- `demucs-mlx` for source separation on MLX.
- `mlx-audio-io` for high-performance native audio I/O.
- CLI entry point `allin1-mlx`.
- In-memory demix + spectrogram pipeline (no intermediate files by default).
- Parallel ensemble inference.
- JSONL timing output (`--timings-path`).
- Selective stage overwrite (`--overwrite`).
- Visualization and sonification outputs.

### Changed

- Renamed package from `allin1` to `allin1_mlx`.
- Requires Python 3.10+ and macOS 13+ on Apple Silicon.
- Default device is `mlx` (no CUDA/CPU path).
- Spectrogram backend defaults to `mlx_fast` (uses demucs-mlx STFT kernels).

### Removed

- PyTorch dependency (torch is no longer required).
- CUDA and CPU inference paths.
- Training code and instructions.
- madmom as a required dependency (optional for comparison only).

[unreleased]: https://github.com/ssmall256/all-in-one-mlx/compare/v1.0.6...HEAD
[1.0.6]: https://github.com/ssmall256/all-in-one-mlx/compare/v1.0.5...v1.0.6
[1.0.5]: https://github.com/ssmall256/all-in-one-mlx/compare/v1.0.4...v1.0.5
[1.0.3]: https://github.com/ssmall256/all-in-one-mlx/compare/v1.0.2...v1.0.3
[1.0.2]: https://github.com/ssmall256/all-in-one-mlx/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/ssmall256/all-in-one-mlx/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/ssmall256/all-in-one-mlx/releases/tag/v1.0.0
