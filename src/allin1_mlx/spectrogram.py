from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
from mlx_audio_io import batch_load as mlx_batch_load
from mlx_audio_io import load as mlx_load_audio
from tqdm import tqdm

# Madmom backend is deprecated - kept only for backwards compatibility
# The default MLX backend is 4-5x faster than madmom
# Madmom imports are lazy-loaded only when explicitly requested via spec_backend='madmom'


_SPEC_FRAME_SIZE = 2048
_SPEC_FPS = int(44100 / 441)
_SPEC_NUM_BANDS = 12
_SPEC_FMIN = 30.0
_SPEC_FMAX = 17000.0
_SPEC_FREF = 440.0

_SPEC_CHECKED = False


def _madmom_processor():
  from madmom.audio.signal import FramedSignalProcessor
  from madmom.audio.spectrogram import FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor
  from madmom.audio.stft import ShortTimeFourierTransformProcessor
  from madmom.processors import SequentialProcessor

  frames = FramedSignalProcessor(
    frame_size=_SPEC_FRAME_SIZE,
    fps=_SPEC_FPS,
  )
  stft = ShortTimeFourierTransformProcessor()
  filt = FilteredSpectrogramProcessor(
    num_bands=_SPEC_NUM_BANDS,
    fmin=_SPEC_FMIN,
    fmax=_SPEC_FMAX,
    norm_filters=True,
  )
  spec = LogarithmicSpectrogramProcessor(mul=1, add=1)
  return SequentialProcessor([frames, stft, filt, spec])


def _load_wave_mono(path: Path) -> Tuple[np.ndarray, int]:
  # mlx-audio-io with built-in mono conversion (11x faster than scipy.io.wavfile)
  data, sr = mlx_load_audio(str(path), mono=True, dtype="float32")
  # Convert MLX array to NumPy and squeeze to 1D
  mono = np.array(data, copy=False).squeeze()
  return mono, sr


def _signal_frame(signal: np.ndarray, index: int, frame_size: int, hop_size: float, origin: int = 0) -> np.ndarray:
  frame_size = int(frame_size)
  num_samples = len(signal)
  ref_sample = int(index * hop_size)
  start = ref_sample - frame_size // 2 - int(origin)
  stop = start + frame_size

  if start >= 0 and stop <= num_samples:
    return signal[start:stop]

  frame = np.repeat(signal[:1], frame_size, axis=0)
  left, right = 0, 0
  if start < 0:
    left = min(stop, 0) - start
    frame[:left] = 0
    start = 0
  if stop > num_samples:
    right = stop - max(start, num_samples)
    frame[-right:] = 0
    stop = num_samples
  frame[left:frame_size - right] = signal[min(start, num_samples):max(stop, 0)]
  return frame


def _frame_signal(signal: np.ndarray, frame_size: int, hop_size: float) -> np.ndarray:
  num_frames = int(np.ceil(len(signal) / float(hop_size)))
  hop_round = int(round(hop_size))
  if abs(hop_size - hop_round) < 1e-6 and hop_round > 0:
    left_pad = frame_size // 2
    right_pad = frame_size // 2 + hop_round
    padded = np.zeros(len(signal) + left_pad + right_pad, dtype=signal.dtype)
    padded[left_pad:left_pad + len(signal)] = signal
    strides = (padded.strides[0] * hop_round, padded.strides[0])
    frames = np.lib.stride_tricks.as_strided(
      padded,
      shape=(num_frames, frame_size),
      strides=strides,
      writeable=False,
    )
    return frames.copy()

  frames = np.empty((num_frames, frame_size), dtype=signal.dtype)
  for idx in range(num_frames):
    frames[idx] = _signal_frame(signal, idx, frame_size, hop_size, origin=0)
  return frames


def _fft_bin_frequencies(num_bins: int, sample_rate: float) -> np.ndarray:
  return np.fft.fftfreq(num_bins * 2, 1.0 / sample_rate)[:num_bins]


def _log_frequencies(bands_per_octave: int, fmin: float, fmax: float, fref: float) -> np.ndarray:
  left = np.floor(np.log2(float(fmin) / fref) * bands_per_octave)
  right = np.ceil(np.log2(float(fmax) / fref) * bands_per_octave)
  frequencies = fref * 2.0 ** (np.arange(left, right) / float(bands_per_octave))
  frequencies = frequencies[np.searchsorted(frequencies, fmin):]
  frequencies = frequencies[:np.searchsorted(frequencies, fmax, 'right')]
  return frequencies


def _frequencies_to_bins(
  frequencies: np.ndarray,
  bin_frequencies: np.ndarray,
  unique_bins: bool,
) -> np.ndarray:
  frequencies = np.asarray(frequencies)
  bin_frequencies = np.asarray(bin_frequencies)
  indices = bin_frequencies.searchsorted(frequencies)
  indices = np.clip(indices, 1, len(bin_frequencies) - 1)
  left = bin_frequencies[indices - 1]
  right = bin_frequencies[indices]
  indices -= frequencies - left < right - frequencies
  if unique_bins:
    indices = np.unique(indices)
  return indices


def _triangular_filters(bins: np.ndarray, norm: bool) -> List[Tuple[np.ndarray, int]]:
  if len(bins) < 3:
    raise ValueError("not enough bins to create a triangular filterbank")
  filters: List[Tuple[np.ndarray, int]] = []
  for idx in range(len(bins) - 2):
    start, center, stop = bins[idx:idx + 3]
    if stop - start < 2:
      center = start
      stop = start + 1
    center = int(center)
    start = int(start)
    stop = int(stop)
    rel_center = center - start
    rel_stop = stop - start
    data = np.zeros(rel_stop, dtype=np.float32)
    if rel_center > 0:
      data[:rel_center] = np.linspace(0, 1, rel_center, endpoint=False)
    data[rel_center:] = np.linspace(1, 0, rel_stop - rel_center, endpoint=False)
    if norm:
      data = data / np.sum(data)
    filters.append((data, start))
  return filters


def _log_filterbank_matrix(
  num_fft_bins: int,
  sample_rate: int,
  num_bands: int,
  fmin: float,
  fmax: float,
  fref: float,
  norm_filters: bool,
  unique_filters: bool,
) -> np.ndarray:
  bin_freqs = _fft_bin_frequencies(num_fft_bins, sample_rate)
  frequencies = _log_frequencies(num_bands, fmin, fmax, fref)
  bins = _frequencies_to_bins(frequencies, bin_freqs, unique_bins=unique_filters)
  filters = _triangular_filters(bins, norm=norm_filters)
  fb = np.zeros((len(bin_freqs), len(filters)), dtype=np.float32)
  for band_idx, (filt, start) in enumerate(filters):
    stop = start + len(filt)
    if start < 0:
      filt = filt[-start:]
      start = 0
    if stop > len(bin_freqs):
      filt = filt[:-(stop - len(bin_freqs))]
      stop = len(bin_freqs)
    if start >= stop:
      continue
    band = fb[start:stop, band_idx]
    np.maximum(filt, band, out=band)
  return fb



@lru_cache(maxsize=8)
def _get_cached_window(sample_rate: int, signal_dtype) -> np.ndarray:
  """Get or create cached window array."""
  window = np.hanning(_SPEC_FRAME_SIZE).astype(np.float32)
  if np.issubdtype(signal_dtype, np.integer):
    window = window / float(np.iinfo(signal_dtype).max)
  return window


@lru_cache(maxsize=8)
def _get_cached_filterbank(sample_rate: int) -> np.ndarray:
  """Get or create cached filterbank matrix."""
  num_fft_bins = _SPEC_FRAME_SIZE >> 1
  fb = _log_filterbank_matrix(
    num_fft_bins=num_fft_bins,
    sample_rate=sample_rate,
    num_bands=_SPEC_NUM_BANDS,
    fmin=_SPEC_FMIN,
    fmax=_SPEC_FMAX,
    fref=_SPEC_FREF,
    norm_filters=True,
    unique_filters=True,
  )
  return fb


@lru_cache(maxsize=8)
def _get_cached_window_mlx(sample_rate: int, signal_dtype):
  """Get or create cached MLX window array."""
  import mlx.core as mx
  window = _get_cached_window(sample_rate, signal_dtype)
  return mx.array(window, dtype=mx.float32)


@lru_cache(maxsize=8)
def _get_cached_filterbank_mlx(sample_rate: int):
  """Get or create cached MLX filterbank matrix."""
  import mlx.core as mx
  fb = _get_cached_filterbank(sample_rate)
  return mx.array(fb, dtype=mx.float32)


def _mlx_log_spectrogram(signal: np.ndarray, sample_rate: int) -> np.ndarray:
  import mlx.core as mx

  frame_size = _SPEC_FRAME_SIZE
  hop_size = sample_rate / float(_SPEC_FPS)
  frames = _frame_signal(signal, frame_size, hop_size)

  frames_mx = mx.array(frames, dtype=mx.float32)
  window_mx = _get_cached_window_mlx(sample_rate, signal.dtype)
  fft_in = frames_mx * window_mx
  stft = mx.fft.fft(fft_in, n=frame_size, axis=1)[:, :frame_size >> 1]
  mag = mx.abs(stft)
  filtered = mag @ _get_cached_filterbank_mlx(sample_rate)
  logged = mx.log10(filtered + 1.0)
  return np.array(logged, copy=False)


def _is_mx_array(value: Any) -> bool:
  try:
    import mlx.core as mx
  except Exception:
    return False
  return isinstance(value, mx.array)


def _mlx_log_spectrogram_batch(
  signals: List[np.ndarray],
  sample_rate: int,
  return_mx: bool = False,
) -> np.ndarray:
  """Batch process multiple signals (stems) together for better performance."""
  import mlx.core as mx

  if not signals:
    raise ValueError("signals must not be empty.")

  # The "mlx" backend uses NumPy framing for madmom parity.
  if _is_mx_array(signals[0]):
    signals = [np.array(signal, copy=False) for signal in signals]

  frame_size = _SPEC_FRAME_SIZE
  hop_size = sample_rate / float(_SPEC_FPS)

  # Frame all signals on CPU (necessary for madmom parity)
  all_frames = []
  for signal in signals:
    frames = _frame_signal(signal, frame_size, hop_size)
    all_frames.append(frames)

  # Stack frames: (num_stems, num_frames, frame_size)
  frames_batch = np.stack(all_frames, axis=0)

  # Transfer to MLX and process all stems at once
  frames_mx = mx.array(frames_batch, dtype=mx.float32)
  window_mx = _get_cached_window_mlx(sample_rate, signals[0].dtype)

  # Broadcast window across all stems
  fft_in = frames_mx * window_mx

  # FFT for all stems at once
  stft = mx.fft.fft(fft_in, n=frame_size, axis=2)[:, :, :frame_size >> 1]
  mag = mx.abs(stft)

  # Apply filterbank to all stems
  filtered = mag @ _get_cached_filterbank_mlx(sample_rate)
  logged = mx.log10(filtered + 1.0)

  if return_mx:
    return logged
  # Force evaluation and return
  mx.eval(logged)
  return np.array(logged, copy=False)


@lru_cache(maxsize=8)
def _get_cached_mlx_stft(sample_rate: int):
  try:
    from demucs_mlx.spec_mlx import SpectralTransform
  except Exception as exc:
    raise ImportError(
      "demucs-mlx is required for mlx_fast spectrograms. Install it with "
      "`uv pip install demucs-mlx`."
    ) from exc

  hop_size = sample_rate / float(_SPEC_FPS)
  hop = int(round(hop_size))
  if abs(hop_size - hop) > 1e-6:
    raise ValueError(f"Non-integer hop size {hop_size} for sample rate {sample_rate}.")
  transform = SpectralTransform(
    n_fft=_SPEC_FRAME_SIZE,
    hop_length=hop,
    win_length=_SPEC_FRAME_SIZE,
    window_fn="hann",
    periodic=False,
    center=False,
    normalized=False,
  )
  return transform, hop


def _mlx_log_spectrogram_fast_batch(
  signals: List[np.ndarray],
  sample_rate: int,
  return_mx: bool = False,
) -> np.ndarray:
  """Fast MLX spectrogram using demucs-mlx STFT kernels."""
  import mlx.core as mx

  if not signals:
    raise ValueError("signals must not be empty.")

  transform, hop = _get_cached_mlx_stft(sample_rate)
  n_fft = _SPEC_FRAME_SIZE
  if _is_mx_array(signals[0]):
    length = int(signals[0].shape[-1])
  else:
    length = len(signals[0])
  num_frames = int(np.ceil(length / float(hop)))
  left_pad = n_fft // 2
  right_pad = n_fft + (num_frames - 1) * hop - length - left_pad
  if right_pad < 0:
    right_pad = 0

  if _is_mx_array(signals[0]):
    x = mx.stack(signals, axis=0).astype(mx.float32)
  else:
    stacked = np.stack(signals).astype(np.float32, copy=False)
    x = mx.array(stacked)
  x = mx.pad(x, [(0, 0), (left_pad, right_pad)], mode="constant")

  spec = transform.stft(x)  # [B, F, N] complex
  spec = spec[:, :-1, :]  # Drop Nyquist to match rfft/fft slicing
  mag = mx.abs(spec)  # [B, F, N]
  mag = mx.transpose(mag, (0, 2, 1))  # [B, N, F]
  filtered = mag @ _get_cached_filterbank_mlx(sample_rate)
  logged = mx.log10(filtered + 1.0)
  if return_mx:
    return logged
  mx.eval(logged)
  return np.array(logged, copy=False)


def _to_mono_signal(signal: np.ndarray) -> np.ndarray:
  if _is_mx_array(signal):
    import mlx.core as mx

    if signal.ndim == 1:
      return signal
    if signal.ndim == 2:
      # Handle (channels, time) or (time, channels)
      if int(signal.shape[0]) in (1, 2) and int(signal.shape[1]) > int(signal.shape[0]):
        return mx.mean(signal, axis=0)
      return mx.mean(signal, axis=1)
    raise ValueError(f"Expected 1D or 2D audio array, got shape {tuple(signal.shape)}.")

  signal_np = np.asarray(signal)
  if signal_np.ndim == 1:
    return signal_np
  if signal_np.ndim == 2:
    # Handle (channels, time) or (time, channels)
    if signal_np.shape[0] in (1, 2) and signal_np.shape[1] > signal_np.shape[0]:
      return signal_np.mean(axis=0)
    return signal_np.mean(axis=1)
  raise ValueError(f"Expected 1D or 2D audio array, got shape {signal_np.shape}.")


def spectrogram_from_stems(
  stems: dict,
  sample_rate: int,
  backend: str = "mlx",
  check: bool = False,
  return_mx: bool = False,
) -> np.ndarray:
  """Compute spectrograms directly from in-memory stems."""
  stem_order = ['bass', 'drums', 'other', 'vocals']
  signals = []
  for stem in stem_order:
    if stem not in stems:
      raise KeyError(f"Missing stem '{stem}' in stems dict.")
    signals.append(_to_mono_signal(stems[stem]))

  if backend == "mlx":
    spec = _mlx_log_spectrogram_batch(signals, sample_rate, return_mx=return_mx)
  elif backend == "mlx_fast":
    try:
      spec = _mlx_log_spectrogram_fast_batch(signals, sample_rate, return_mx=return_mx)
    except ImportError as exc:
      print(f"=> mlx_fast unavailable ({exc}); falling back to mlx.")
      spec = _mlx_log_spectrogram_batch(signals, sample_rate, return_mx=return_mx)
  else:
    raise ValueError(f"Unknown spectrogram backend '{backend}'.")

  if check:
    global _SPEC_CHECKED
    if not _SPEC_CHECKED:
      ref = _mlx_log_spectrogram_batch(signals, sample_rate)
      spec_np = np.array(spec, copy=False) if return_mx else spec
      diff = np.abs(spec_np - ref)
      print(f"=> Spectrogram check (mlx_fast vs mlx): max={diff.max():.6f}, mean={diff.mean():.6f}")
      _SPEC_CHECKED = True
  return spec


def extract_spectrograms(
  demix_paths: List[Path],
  spec_dir: Path,
  multiprocess: bool = True,
  overwrite: bool = False,
  backend: str = "madmom",
  check: bool = False,
):
  todos = []
  spec_paths = []
  for src in demix_paths:
    dst = spec_dir / f'{src.name}.npy'
    spec_paths.append(dst)
    if dst.is_file():
      if overwrite:
        dst.unlink(missing_ok=True)
      else:
        continue
    todos.append((src, dst))

  if overwrite:
    existing = 0
  else:
    existing = len(spec_paths) - len(todos)
  print(f'=> Found {existing} spectrograms already extracted, {len(todos)} to extract.')

  if todos:
    if backend == "madmom":
      processor = _madmom_processor()
      if multiprocess:
        pool = Pool()
        map_fn = pool.imap
      else:
        pool = None
        map_fn = map

      iterator = map_fn(_extract_spectrogram_madmom, [
        (src, dst, processor)
        for src, dst in todos
      ])
      for _ in tqdm(iterator, total=len(todos), desc='Extracting spectrograms'):
        pass

      if pool:
        pool.close()
        pool.join()
    else:
      if multiprocess:
        print("=> Multiprocessing disabled for mlx spectrogram backends.")
      iterator = map(_extract_spectrogram_backend, [
        (src, dst, backend, check)
        for src, dst in todos
      ])
      for _ in tqdm(iterator, total=len(todos), desc='Extracting spectrograms'):
        pass

  return spec_paths


def _extract_spectrogram_madmom(args):
  from madmom.audio.signal import Signal

  src, dst, processor = args

  dst.parent.mkdir(parents=True, exist_ok=True)

  sig_bass = Signal(src / 'bass.wav', num_channels=1)
  sig_drums = Signal(src / 'drums.wav', num_channels=1)
  sig_other = Signal(src / 'other.wav', num_channels=1)
  sig_vocals = Signal(src / 'vocals.wav', num_channels=1)

  spec_bass = processor(sig_bass)
  spec_drums = processor(sig_drums)
  spec_others = processor(sig_other)
  spec_vocals = processor(sig_vocals)

  spec = np.stack([spec_bass, spec_drums, spec_others, spec_vocals])  # instruments, frames, bins

  np.save(str(dst), spec)


def _extract_spectrogram_backend(args: Tuple[Path, Path, str, bool]):
  src, dst, backend, check = args

  dst.parent.mkdir(parents=True, exist_ok=True)

  # Load all 4 stems in parallel using batch_load (~4x faster than sequential)
  stem_names = ['bass', 'drums', 'other', 'vocals']
  stem_paths = [str(src / f'{stem}.wav') for stem in stem_names]
  results = mlx_batch_load(stem_paths, mono=True, dtype="float32", num_workers=4)
  stems = [(np.array(audio, copy=False).squeeze(), sr) for audio, sr in results]

  if backend == "mlx":
    sr = stems[0][1]
    spec = _mlx_log_spectrogram_batch([signal for signal, _ in stems], sr)
  elif backend == "mlx_fast":
    sr = stems[0][1]
    try:
      spec = _mlx_log_spectrogram_fast_batch([signal for signal, _ in stems], sr)
    except ImportError as exc:
      print(f"=> mlx_fast unavailable ({exc}); falling back to mlx.")
      spec = _mlx_log_spectrogram_batch([signal for signal, _ in stems], sr)
    if check:
      global _SPEC_CHECKED
      if not _SPEC_CHECKED:
        ref = _mlx_log_spectrogram_batch([signal for signal, _ in stems], sr)
        diff = np.abs(spec - ref)
        print(f"=> Spectrogram check (mlx_fast vs mlx): max={diff.max():.6f}, mean={diff.mean():.6f}")
        _SPEC_CHECKED = True
  else:
    raise ValueError(f"Unknown spectrogram backend '{backend}'.")
  np.save(str(dst), spec)
