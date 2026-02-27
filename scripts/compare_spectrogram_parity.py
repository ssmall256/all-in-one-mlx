#!/usr/bin/env python
"""Compare madmom spectrograms against a torch implementation and report parity."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

try:
    from madmom.audio.signal import FramedSignalProcessor, Signal
    from madmom.audio.spectrogram import FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor
    from madmom.audio.stft import ShortTimeFourierTransformProcessor
    from madmom.processors import SequentialProcessor
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "madmom is required for this script. Install it in the current environment."
    ) from exc


STEMS = ["bass", "drums", "other", "vocals"]


def _make_madmom_processor() -> SequentialProcessor:
    frames = FramedSignalProcessor(
        frame_size=2048,
        fps=int(44100 / 441),
    )
    stft = ShortTimeFourierTransformProcessor()
    filt = FilteredSpectrogramProcessor(
        num_bands=12,
        fmin=30,
        fmax=17000,
        norm_filters=True,
    )
    spec = LogarithmicSpectrogramProcessor(mul=1, add=1)
    return SequentialProcessor([frames, stft, filt, spec])


def _infer_log_base(log_proc: LogarithmicSpectrogramProcessor) -> str:
    log_fn = getattr(log_proc, "log", None)
    if log_fn is np.log10:
        return "log10"
    if log_fn is np.log:
        return "ln"
    return "ln"


def _extract_filterbank_matrix(filt: FilteredSpectrogramProcessor) -> np.ndarray:
    fb = None
    for attr in ("filterbank", "filters", "fb"):
        if hasattr(filt, attr):
            fb = getattr(filt, attr)
            break
    if fb is None and hasattr(filt, "filterbank"):
        fb = filt.filterbank
    if fb is None:
        raise RuntimeError("Unable to locate filterbank matrix on FilteredSpectrogramProcessor.")
    if hasattr(fb, "filters"):
        fb = fb.filters
    fb = np.asarray(fb)
    if fb.ndim != 2:
        raise RuntimeError("Filterbank matrix must be 2D.")
    return fb


def _madmom_frames_and_window(
    path: Path,
    frames_proc: FramedSignalProcessor,
    stft_proc: ShortTimeFourierTransformProcessor,
) -> Tuple[np.ndarray, int, np.ndarray, int, bool, bool]:
    sig = Signal(str(path), num_channels=1)
    frames = frames_proc(sig)
    frames_np = np.asarray(frames)
    if frames_np.ndim == 3:
        frames_np = np.mean(frames_np, axis=-1)
    frame_size = frames_np.shape[1]
    window = stft_proc.window
    if callable(window):
        window = window(frame_size)
    if window is None:
        window = np.ones(frame_size)
    try:
        max_range = float(np.iinfo(frames.signal.dtype).max)
        window = window / max_range
    except ValueError:
        pass
    fft_size = stft_proc.fft_size or frame_size
    include_nyquist = bool(stft_proc.include_nyquist)
    circular_shift = bool(stft_proc.circular_shift)
    return frames_np, frames.signal.sample_rate, np.asarray(window), int(fft_size), include_nyquist, circular_shift


def _torch_spectrogram(
    frames_np: np.ndarray,
    window: np.ndarray,
    fft_size: int,
    include_nyquist: bool,
    circular_shift: bool,
    filt: FilteredSpectrogramProcessor,
    log_proc: LogarithmicSpectrogramProcessor,
    device: torch.device,
    dtype: torch.dtype,
    power: int,
    log_base: str,
) -> np.ndarray:
    fb = _extract_filterbank_matrix(filt)

    frames_t = torch.tensor(frames_np, device=device, dtype=dtype)
    window_t = torch.tensor(window, device=device, dtype=dtype)
    if circular_shift:
        fft_shift = frames_np.shape[1] >> 1
        frames_t = frames_t * window_t
        zeros = torch.zeros((frames_t.shape[0], fft_size), device=device, dtype=dtype)
        zeros[:, :fft_shift] = frames_t[:, fft_shift:]
        zeros[:, -fft_shift:] = frames_t[:, :fft_shift]
        fft_in = zeros
    else:
        fft_in = frames_t * window_t

    stft_out = torch.fft.fft(fft_in, n=fft_size, dim=1)
    num_fft_bins = (fft_size >> 1) + (1 if include_nyquist else 0)
    stft_out = stft_out[:, :num_fft_bins]
    mag = stft_out.abs()
    if power == 2:
        mag = mag * mag

    fb_t = torch.tensor(fb, device=device, dtype=dtype)
    if fb_t.shape[0] == mag.shape[1]:
        filtered = mag @ fb_t
    elif fb_t.shape[1] == mag.shape[1]:
        filtered = mag @ fb_t.transpose(0, 1)
    else:
        raise RuntimeError(
            f"Filterbank shape {fb_t.shape} incompatible with mag shape {mag.shape}."
        )

    mul = float(getattr(log_proc, "mul", 1.0))
    add = float(getattr(log_proc, "add", 0.0))
    if log_base == "log10":
        logged = torch.log10(filtered * mul + add)
    else:
        logged = torch.log(filtered * mul + add)

    return logged.detach().cpu().numpy()


def _madmom_spectrogram(
    path: Path,
    processor: SequentialProcessor,
) -> np.ndarray:
    sig = Signal(str(path), num_channels=1)
    return processor(sig)


def _iter_tracks(demix_dir: Path, limit: int | None) -> Iterable[Path]:
    roots = sorted(demix_dir.glob("*/"))
    for root in roots[: limit or None]:
        yield root


def _metrics(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    if a.shape != b.shape:
        min_frames = min(a.shape[0], b.shape[0])
        min_bins = min(a.shape[1], b.shape[1])
        a = a[:min_frames, :min_bins]
        b = b[:min_frames, :min_bins]
    diff = a - b
    return {
        "mean_abs": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "mean_rel": float(np.mean(np.abs(diff) / (np.abs(a) + 1e-8))),
    }


def _aggregate(metrics: List[Dict[str, float]]) -> Dict[str, float]:
    keys = metrics[0].keys() if metrics else []
    return {k: float(np.mean([m[k] for m in metrics])) for k in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("demix_dir", type=Path, help="Path to htdemucs output dir (containing track folders).")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tracks to sample.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device.")
    parser.add_argument("--dtype", type=str, default="float32", help="Torch dtype.")
    parser.add_argument(
        "--center",
        action="store_true",
        help="Unused (framing follows madmom). Kept for compatibility.",
    )
    parser.add_argument("--power", type=int, choices=[1, 2], default=1, help="Magnitude power (1=mag, 2=power).")
    parser.add_argument("--log-base", type=str, choices=["ln", "log10", "auto"], default="auto")
    parser.add_argument(
        "--search",
        action="store_true",
        help="Grid search over center/power/log-base and report the best parity.",
    )
    parser.add_argument(
        "--no-per-stem",
        action="store_true",
        help="Suppress per-stem metrics output (useful for grid search).",
    )
    args = parser.parse_args()

    processor = _make_madmom_processor()
    frames, stft, filt, log_proc = processor
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    def _run_config(power: int, log_base: str) -> Dict[str, float]:
        all_metrics: List[Dict[str, float]] = []
        for track_dir in _iter_tracks(args.demix_dir, args.limit):
            for stem in STEMS:
                wav_path = track_dir / f"{stem}.wav"
                if not wav_path.is_file():
                    continue
                madmom_spec = _madmom_spectrogram(wav_path, processor)
                frames_np, _sr, window, fft_size, include_nyquist, circular_shift = _madmom_frames_and_window(
                    wav_path, frames, stft
                )
                torch_spec = _torch_spectrogram(
                    frames_np=frames_np,
                    window=window,
                    fft_size=fft_size,
                    include_nyquist=include_nyquist,
                    circular_shift=circular_shift,
                    filt=filt,
                    log_proc=log_proc,
                    device=device,
                    dtype=dtype,
                    power=power,
                    log_base=log_base,
                )
                metrics = _metrics(madmom_spec, torch_spec)
                all_metrics.append(metrics)
                if not args.no_per_stem:
                    print(
                        f"{track_dir.name}/{stem}: "
                        f"mean_abs={metrics['mean_abs']:.6f} "
                        f"max_abs={metrics['max_abs']:.6f} "
                        f"rmse={metrics['rmse']:.6f} "
                        f"mean_rel={metrics['mean_rel']:.6f}"
                    )
        if not all_metrics:
            return {}
        return _aggregate(all_metrics)

    if args.search:
        configs: List[Tuple[int, str]] = []
        for power in (1, 2):
            for log_base in ("ln", "log10"):
                configs.append((power, log_base))

        results: List[Tuple[Tuple[int, str], Dict[str, float]]] = []
        for power, log_base in configs:
            if args.log_base != "auto" and log_base != args.log_base:
                continue
            if not args.no_per_stem:
                print(f"== config power={power} log_base={log_base} ==")
            agg = _run_config(power, log_base)
            if agg:
                results.append(((power, log_base), agg))
                print(
                    "overall: "
                    f"mean_abs={agg['mean_abs']:.6f} "
                    f"max_abs={agg['max_abs']:.6f} "
                    f"rmse={agg['rmse']:.6f} "
                    f"mean_rel={agg['mean_rel']:.6f}"
                )

        if results:
            best = min(results, key=lambda item: item[1]["mean_abs"])
            (power, log_base), agg = best
            print(
                "best: "
                f"power={power} log_base={log_base} "
                f"mean_abs={agg['mean_abs']:.6f} "
                f"max_abs={agg['max_abs']:.6f} "
                f"rmse={agg['rmse']:.6f} "
                f"mean_rel={agg['mean_rel']:.6f}"
            )
        return

    log_base = args.log_base
    if log_base == "auto":
        log_base = _infer_log_base(log_proc)
    agg = _run_config(args.power, log_base)
    if agg:
        print(
            "overall: "
            f"mean_abs={agg['mean_abs']:.6f} "
            f"max_abs={agg['max_abs']:.6f} "
            f"rmse={agg['rmse']:.6f} "
            f"mean_rel={agg['mean_rel']:.6f}"
        )


if __name__ == "__main__":
    main()
