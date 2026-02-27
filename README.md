# all-in-one-mlx

**GPU-accelerated music structure analysis on Apple Silicon (MLX).**

`all-in-one-mlx` is an Apple Silicon–optimized port of the original **All‑In‑One Music Structure Analyzer** (upstream: [`mir-aidj/all-in-one`](https://github.com/mir-aidj/all-in-one)). It runs end‑to‑end inference locally using **Apple MLX**, with an integrated pipeline designed for real songs (including demixing + fast spectrograms).

Given one or more audio tracks, it produces:

- **Tempo (BPM)**
- **Beat times**
- **Downbeat times**
- **Section boundaries**
- **Section labels** (intro / verse / chorus / bridge / outro, etc.)

---

## Why this repo exists

The upstream project is a strong reference implementation, but macOS Apple Silicon users historically lacked a first‑class GPU‑accelerated inference path. This repository provides that acceleration via **MLX**, with an emphasis on:

- **High performance** on M‑series GPUs
- **Practical CLI defaults** for song inference (demix → spectrogram → model → outputs)
- **Faithful behavior** to the upstream model + method

I’m releasing **all-in-one-mlx** alongside [**all-in-one-mps**](https://github.com/ssmall256/all-in-one-mps) (PyTorch/MPS acceleration) so Apple Silicon users can choose the stack that fits their environment.

---

## Performance

Benchmark on a single file — Apple M4 Max, 128 GB RAM, macOS 26.3:

| Project | Time | vs upstream |
|---|---|---|
| [`mir-aidj/all-in-one`](https://github.com/mir-aidj/all-in-one) | 75.25s | baseline |
| [`mir-aidj/all-in-one`](https://github.com/mir-aidj/all-in-one) | 24.63s | ~3.1x faster *(patched to use MPS)* |
| [`all-in-one-mps`](https://github.com/ssmall256/all-in-one-mps) | 13.43s | ~5.6x faster |
| **`all-in-one-mlx` (this repo)** | **5.96s** | **~12.6x faster** |

> One run, one file — results will vary by hardware and content.

---

## Related projects & attribution

| Project | Purpose |
|---|---|
| [`mir-aidj/all-in-one`](https://github.com/mir-aidj/all-in-one) | Original reference implementation and training code |
| [`all-in-one-mlx`](https://github.com/ssmall256/all-in-one-mlx) | This repo: MLX inference + packaging for Apple Silicon |
| [`all-in-one-mps`](https://github.com/ssmall256/all-in-one-mps) | Companion repo: PyTorch/MPS inference for Apple Silicon |

This repository began as a fork/port of the upstream project. The original method/model is described in:

- Taejun Kim & Juhan Nam, *All‑In‑One Metrical and Functional Structure Analysis with Neighborhood Attentions on Demixed Audio* ([arXiv:2307.16425](https://arxiv.org/abs/2307.16425))

If you use this in academic work, please cite the paper and the [upstream repository](https://github.com/mir-aidj/all-in-one).

---

## Requirements

| Component | Requirement |
|---|---|
| Hardware | Apple Silicon (M-series) |
| OS | macOS 14+ (required by MLX wheels) |
| Python | 3.10+ |

> Need CUDA / Linux / Windows? Use the [upstream project](https://github.com/mir-aidj/all-in-one).

---

## Installation

### pip

```bash
pip install all-in-one-mlx
```

### uv (recommended)

```bash
uv pip install all-in-one-mlx
```

---

## Quickstart

Analyze one or more tracks:

```bash
allin1-mlx path/to/song.wav
# or multiple:
allin1-mlx path/to/a.wav path/to/b.wav
```

By default, results are written under:

- `./struct` (set with `--out-dir`)

### Common options

- Choose output directory:

```bash
allin1-mlx song.wav --out-dir ./struct
```

- Save visualizations / sonifications:

```bash
allin1-mlx song.wav --visualize --viz-dir ./viz
allin1-mlx song.wav --sonify --sonif-dir ./sonif
```

- Keep intermediate byproducts (demixed audio + spectrograms):

```bash
allin1-mlx song.wav --keep-byproducts
# demix files: ./demix (override with --demix-dir)
# specs:       ./spec  (override with --spec-dir)
```

- Fast spectrogram backend (default) vs reference backend:

```bash
allin1-mlx song.wav --spec-backend mlx_fast   # default
allin1-mlx song.wav --spec-backend mlx        # reference path
```

- One-time spectrogram correctness check (reports max/mean diff):

```bash
allin1-mlx song.wav --spec-check
```

- Overwrite specific stages (demix,spec,json,viz,sonify) or everything:

```bash
allin1-mlx song.wav --overwrite all
allin1-mlx song.wav --overwrite demix,spec,json
```

- Timing / performance instrumentation:

```bash
allin1-mlx song.wav --timings-path timings.jsonl
allin1-mlx song.wav --timings-path timings.jsonl --timings-viz-path timings.png
```

---

## MLX inference controls

- Select model (pretrained name):

```bash
allin1-mlx song.wav --model harmonix-all
```

- Batch size:

```bash
allin1-mlx song.wav --mlx-batch-size 1
```

- `mx.compile` for model forward (enabled by default):

```bash
allin1-mlx song.wav --no-mlx-compile
```

- In-memory pipeline for demix + spectrograms (enabled by default):

```bash
allin1-mlx song.wav --no-mlx-in-memory
```

- Ensemble inference parallelism (enabled by default):

```bash
allin1-mlx song.wav --no-ensemble-parallel
```

- Disable multiprocessing (debug / determinism / constrained envs):

```bash
allin1-mlx song.wav --no-multiprocess
```

---

## Outputs

The CLI writes analysis artifacts under `--out-dir` (default `./struct`). Exact filenames may vary by model/pipeline version, but outputs include tempo, beats, downbeats, and segment boundaries/labels in machine-readable form.

Optional outputs:

| Artifact | Enable with |
|---|---|
| Visualizations | `--visualize` and `--viz-dir` |
| Sonifications | `--sonify` and `--sonif-dir` |
| Frame-level activations | `--activ` |
| Frame-level embeddings | `--embed` |
| JSONL timings | `--timings-path` |

## Model weights

| Item | Behavior |
|---|---|
| Source | MLX checkpoints are loaded from local files |
| Packaging | Release wheels/sdists do not bundle model weights |
| Default lookup path | `./mlx-weights` |
| Custom paths | Use `--mlx-weights-dir` or explicit `--mlx-weights-path` and `--mlx-config-path` |

## Known limitations

- Artifact naming uses input basename/stem for intermediate and output files.
- If multiple inputs share the same basename (for example `a/song.mp3` and `b/song.wav`), artifacts may overwrite each other or be reused unexpectedly.
- Workaround: process those files separately or rename files so basenames are unique.

---

## License

This project retains the upstream license (MIT). See `LICENSE`.

---

## Issues

Please include:

| Include | Example |
|---|---|
| macOS version + Apple Silicon model | `macOS 26.3, M4 Max` |
| Python + MLX versions | `Python 3.12.7, mlx x.y` |
| Exact command and logs/traceback | Full `allin1-mlx ...` command + stack trace |
