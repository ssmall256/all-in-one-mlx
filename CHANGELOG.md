# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[unreleased]: https://github.com/ssmall256/all-in-one-mlx/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/ssmall256/all-in-one-mlx/releases/tag/v1.0.0
