from typing import Union

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray
from scipy.signal import argrelextrema

from ..config import Config


def event_frames_to_time(
  tensor: Union[NDArray, np.ndarray],
  cfg: Config = None,
  sample_rate: int = None,
  hop_size: int = None,
):
  """
  Args:
    tensor: a binary event tensor with shape (batch, frame)
  """
  assert len(tensor.shape) in (1, 2), 'Input tensor should have 1 or 2 dimensions'

  if cfg is not None:
    sample_rate = cfg.sample_rate
    hop_size = cfg.hop_size

  original_shape = tensor.shape
  if len(original_shape) == 1:
    tensor = tensor[None, :]

  batch_size = tensor.shape[0]
  i_examples, i_frames = np.where(tensor)
  times = i_frames * hop_size / sample_rate
  times = [times[i_examples == i] for i in range(batch_size)]

  if len(original_shape) == 1:
    times = times[0]
  return times


def local_maxima_numpy(arr, order=20):
  is_batch = len(arr.shape) == 2
  if is_batch:
    return np.stack([local_maxima_numpy(x, order) for x in arr])

  compare_func = np.greater
  local_maxima_indices = argrelextrema(arr, compare_func, order=order)

  output_arr = np.zeros_like(arr)
  output_arr[local_maxima_indices] = arr[local_maxima_indices]

  return output_arr


def local_maxima_numpy_window(arr, filter_size=41):
  assert len(arr.shape) in (1, 2), 'Input array should have 1 or 2 dimensions'
  assert filter_size % 2 == 1, 'Filter size should be an odd number'

  original_shape = arr.shape
  if len(original_shape) == 1:
    arr = arr[None, :]

  padding = filter_size // 2
  padded_arr = np.pad(arr, ((0, 0), (padding, padding)), mode='constant', constant_values=-np.inf)
  rolling_view = sliding_window_view(padded_arr, filter_size, axis=1)

  center = filter_size // 2
  local_maxima_mask = rolling_view[:, :, center] == np.max(rolling_view, axis=-1)
  local_maxima_indices = np.argwhere(local_maxima_mask)

  output_arr = np.zeros_like(arr)
  output_arr[local_maxima_mask] = arr[local_maxima_mask]
  output_arr = output_arr.reshape(original_shape)

  return output_arr, local_maxima_indices


def estimate_tempo_from_beats(pred_beat_times):
  beat_interval = np.diff(pred_beat_times)
  bpm = 60. / beat_interval
  bpm = bpm.round()
  bincount = np.bincount(bpm.astype(int))
  bpm_range = np.arange(len(bincount))
  bpm_strength = bincount / bincount.sum()
  bpm_est = np.stack([bpm_range, bpm_strength], axis=-1)
  bpm_est = bpm_est[np.argsort(bpm_strength)[::-1]]
  bpm_est = bpm_est[bpm_est[:, 1] > 0]
  return bpm_est


def peak_picking(boundary_activation, window_past=12, window_future=6):
  window_size = window_past + window_future
  assert window_size % 2 == 0, 'window_past + window_future must be even'
  window_size += 1

  boundary_activation_padded = np.pad(boundary_activation, (window_past, window_future), mode='constant')
  max_filter = sliding_window_view(boundary_activation_padded, window_size)
  local_maxima = (boundary_activation == np.max(max_filter, axis=-1)) & (boundary_activation > 0)

  past_window_filter = sliding_window_view(boundary_activation_padded[:-(window_future + 1)], window_past)
  future_window_filter = sliding_window_view(boundary_activation_padded[window_past + 1:], window_future)
  past_mean = np.mean(past_window_filter, axis=-1)
  future_mean = np.mean(future_window_filter, axis=-1)
  strength_values = boundary_activation - ((past_mean + future_mean) / 2)

  boundary_candidates = np.flatnonzero(local_maxima)
  strength_values = strength_values[boundary_candidates]

  strength_activations = np.zeros_like(boundary_activation)
  strength_activations[boundary_candidates] = strength_values

  return strength_activations
