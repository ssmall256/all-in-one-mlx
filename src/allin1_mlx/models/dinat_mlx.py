"""MLX port of dinat.py from mir-aidj/all-in-one.

Original: https://github.com/mir-aidj/all-in-one/blob/main/src/allin1/models/dinat.py

Uses natten-mlx extras fused Metal QK+RPB kernels for DiNAT attention.
Layout: spatial-first [B, ..., H, D] — transposition handled inside natten-mlx.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from natten_mlx.extras.allin1 import na1d_av_fused, na1d_qk_rpb, na2d_av_fused, na2d_qk_rpb

from ..config import Config
from .utils_mlx import get_activation_function

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def drop_path(input, drop_prob=0.0, training=False, scale_by_keep=True):
  if drop_prob == 0.0 or not training:
    return input
  keep_prob = 1 - drop_prob
  shape = (input.shape[0],) + (1,) * (input.ndim - 1)
  random_tensor = keep_prob + mx.random.uniform(shape=shape, low=0.0, high=1.0)
  random_tensor = mx.floor(random_tensor)
  if scale_by_keep:
    return input / keep_prob * random_tensor
  return input * random_tensor


class DinatDropPath(nn.Module):
  def __init__(self, drop_prob: Optional[float] = None) -> None:
    super().__init__()
    self.drop_prob = drop_prob

  def __call__(self, hidden_states: mx.array) -> mx.array:
    return drop_path(hidden_states, self.drop_prob, self.training)

  def _extra_repr(self) -> str:
    return f"p={self.drop_prob}"


# ---------------------------------------------------------------------------
# Neighborhood Attention (spatial-first: [B, ..., H, D])
# ---------------------------------------------------------------------------

class _NeighborhoodAttentionNd(ABC, nn.Module):
  rpb: mx.array

  def __init__(
    self,
    cfg: Config,
    dim: int,
    num_heads: int,
    kernel_size: int,
    dilation: int
  ):
    super().__init__()
    if dim % num_heads != 0:
      raise ValueError(
        f"The hidden size ({dim}) is not a multiple of the number of attention heads ({num_heads})"
      )

    self.num_attention_heads = num_heads
    self.attention_head_size = int(dim / num_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size
    self.kernel_size = kernel_size
    self.dilation = dilation
    self.scale = self.attention_head_size ** -0.5

    self.query = nn.Linear(self.all_head_size, self.all_head_size, bias=cfg.qkv_bias)
    self.key = nn.Linear(self.all_head_size, self.all_head_size, bias=cfg.qkv_bias)
    self.value = nn.Linear(self.all_head_size, self.all_head_size, bias=cfg.qkv_bias)

    self.dropout = nn.Dropout(cfg.drop_attention)

  @abstractmethod
  def _qk_rpb(self, q: mx.array, k: mx.array) -> mx.array:
    raise NotImplementedError

  @abstractmethod
  def _av(self, attn: mx.array, v: mx.array) -> mx.array:
    raise NotImplementedError

  def __call__(
    self,
    hidden_states: mx.array,
    output_attentions: Optional[bool] = False,
  ) -> Tuple[mx.array]:
    q = self._reshape_to_heads(self.query(hidden_states))
    k = self._reshape_to_heads(self.key(hidden_states))
    v = self._reshape_to_heads(self.value(hidden_states))

    attention_scores = self._qk_rpb(q, k)
    attention_probs = mx.softmax(attention_scores, axis=-1)
    attention_probs = self.dropout(attention_probs)
    context_layer = self._av(attention_probs, v)

    new_shape = context_layer.shape[:-2] + (self.all_head_size,)
    context_layer = mx.reshape(context_layer, new_shape)

    outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
    return outputs

  def _reshape_to_heads(self, x: mx.array) -> mx.array:
    """[B, ..., C] -> [B, ..., H, D] (spatial-first, no transpose)."""
    new_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
    return mx.reshape(x, new_shape)


class NeighborhoodAttention1d(_NeighborhoodAttentionNd):
  def __init__(self, cfg, dim, num_heads, kernel_size, dilation):
    super().__init__(cfg, dim, num_heads, kernel_size, dilation)
    self.rpb = mx.zeros((num_heads, (2 * kernel_size - 1)))

  def _qk_rpb(self, q, k):
    return na1d_qk_rpb(q, k, self.rpb, self.kernel_size, self.dilation, scale=self.scale)

  def _av(self, attn, v):
    return na1d_av_fused(attn, v, self.kernel_size, self.dilation)


class NeighborhoodAttention2d(_NeighborhoodAttentionNd):
  def __init__(self, cfg, dim, num_heads, kernel_size, dilation):
    super().__init__(cfg, dim, num_heads, kernel_size, dilation)
    self.rpb = mx.zeros((num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))

  def _qk_rpb(self, q, k):
    return na2d_qk_rpb(q, k, self.rpb, self.kernel_size, self.dilation, scale=self.scale)

  def _av(self, attn, v):
    return na2d_av_fused(attn, v, self.kernel_size, self.dilation)


# ---------------------------------------------------------------------------
# Attention output / module wrappers
# ---------------------------------------------------------------------------

class NeighborhoodAttentionOutput(nn.Module):
  def __init__(self, config: Config, dim: int):
    super().__init__()
    self.dense = nn.Linear(dim, dim)
    self.dropout = nn.Dropout(config.drop_attention)

  def __call__(self, hidden_states: mx.array) -> mx.array:
    hidden_states = self.dense(hidden_states)
    hidden_states = self.dropout(hidden_states)
    return hidden_states


class _NeighborhoodAttentionModuleNd(ABC, nn.Module):
  self: _NeighborhoodAttentionNd

  def __init__(self, cfg: Config, dim: int):
    super().__init__()
    self.output = NeighborhoodAttentionOutput(cfg, dim)

  def __call__(
    self,
    hidden_states: mx.array,
    output_attentions: Optional[bool] = False,
  ) -> Tuple[mx.array]:
    self_outputs = self.self(hidden_states, output_attentions)
    attention_output = self.output(self_outputs[0])
    outputs = (attention_output,) + self_outputs[1:]
    return outputs


class NeighborhoodAttentionModule1d(_NeighborhoodAttentionModuleNd):
  def __init__(self, cfg, dim, num_heads, kernel_size, dilation):
    super().__init__(cfg, dim)
    self.self = NeighborhoodAttention1d(cfg, dim, num_heads, kernel_size, dilation)


class NeighborhoodAttentionModule2d(_NeighborhoodAttentionModuleNd):
  def __init__(self, cfg, dim, num_heads, kernel_size, dilation):
    super().__init__(cfg, dim)
    self.self = NeighborhoodAttention2d(cfg, dim, num_heads, kernel_size, dilation)


# ---------------------------------------------------------------------------
# Transformer blocks
# ---------------------------------------------------------------------------

class DinatIntermediate(nn.Module):
  def __init__(self, config: Config, dim_in: int, dim_out: int):
    super().__init__()
    self.dense = nn.Linear(dim_in, dim_out)
    if isinstance(config.act_transformer, str):
      self.intermediate_act_fn = get_activation_function(config.act_transformer)
    else:
      self.intermediate_act_fn = config.act_transformer

  def __call__(self, hidden_states: mx.array) -> mx.array:
    hidden_states = self.dense(hidden_states)
    hidden_states = self.intermediate_act_fn(hidden_states)
    return hidden_states


class DinatOutput(nn.Module):
  def __init__(self, config: Config, dim_in: int, dim_out: int):
    super().__init__()
    self.dense = nn.Linear(dim_in, dim_out)
    self.dropout = nn.Dropout(config.drop_hidden)

  def __call__(self, hidden_states: mx.array) -> mx.array:
    hidden_states = self.dense(hidden_states)
    hidden_states = self.dropout(hidden_states)
    return hidden_states


class _DinatLayerNd(ABC, nn.Module):
  attention: _NeighborhoodAttentionModuleNd
  attention2: Optional[_NeighborhoodAttentionModuleNd]

  def __init__(
    self,
    cfg: Config,
    dim: int,
    kernel_size: int,
    dilation: int,
    drop_path_rate: float,
    double_attention: bool,
  ):
    super().__init__()
    self.double_attention = double_attention
    self.kernel_size = kernel_size
    self.dilation = dilation
    self.window_size = self.kernel_size * self.dilation
    if double_attention:
      self.window_size *= 2
    self.layernorm_before = nn.LayerNorm(dim, eps=cfg.layer_norm_eps)
    self.drop_path = DinatDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
    dim_after = dim * 2 if double_attention else dim
    self.layernorm_after = nn.LayerNorm(dim_after, eps=cfg.layer_norm_eps)
    self.intermediate = DinatIntermediate(cfg, dim_after, int(dim_after * cfg.mlp_ratio))
    self.output = DinatOutput(cfg, int(dim_after * cfg.mlp_ratio), dim)

  @abstractmethod
  def maybe_pad(self, *args, **kwargs):
    raise NotImplementedError

  def __call__(
    self,
    hidden_states: mx.array,
    output_attentions: Optional[bool] = False,
  ) -> Tuple[mx.array]:
    if len(hidden_states.shape) > 3:
      is_2d = True
      n, k, t, c = hidden_states.shape
    else:
      is_2d = False
      n, t, c = hidden_states.shape
    shortcut = hidden_states

    hidden_states = self.layernorm_before(hidden_states)
    if is_2d:
      hidden_states, pad_values = self.maybe_pad(hidden_states, k, t)
    else:
      hidden_states, pad_values = self.maybe_pad(hidden_states, t)

    attention_inputs = hidden_states
    hidden_states_list = []
    for attention in [self.attention, self.attention2]:
      if attention is None:
        continue

      attention_output = attention(attention_inputs, output_attentions=output_attentions)
      attention_output = attention_output[0]

      if is_2d:
        was_padded = pad_values[3] > 0 or pad_values[5] > 0
        if was_padded:
          attention_output = attention_output[:, :k, :t, :]
      else:
        was_padded = pad_values[3] > 0
        if was_padded:
          attention_output = attention_output[:, :t, :]

      hidden_states = shortcut + self.drop_path(attention_output)
      hidden_states_list.append(hidden_states)

    if self.double_attention:
      hidden_states = mx.concatenate(hidden_states_list, axis=-1)
      shortcut = mx.stack(hidden_states_list, axis=0).mean(axis=0)
    else:
      shortcut = hidden_states
    layer_output = self.layernorm_after(hidden_states)
    layer_output = self.output(self.intermediate(layer_output))

    layer_output = shortcut + self.drop_path(layer_output)
    layer_outputs = (layer_output,)
    return layer_outputs


class DinatLayer1d(_DinatLayerNd):
  def __init__(
    self,
    cfg: Config,
    dim: int,
    num_heads: int,
    kernel_size: int,
    dilation: int,
    drop_path_rate: float,
    double_attention: bool,
  ):
    super().__init__(cfg, dim, kernel_size, dilation, drop_path_rate, double_attention)
    self.attention = NeighborhoodAttentionModule1d(cfg, dim, num_heads, kernel_size, dilation)
    if double_attention:
      self.attention2 = NeighborhoodAttentionModule1d(cfg, dim, num_heads, kernel_size, dilation * 2)
    else:
      self.attention2 = None

  def maybe_pad(self, hidden_states, frames):
    window_size = self.window_size
    pad_values = (0, 0, 0, 0)
    if frames < window_size:
      pad_l = 0
      pad_r = max(0, window_size - frames)
      pad_values = (0, 0, pad_l, pad_r)
      hidden_states = mx.pad(hidden_states, ((0, 0), (pad_l, pad_r), (0, 0)))
    return hidden_states, pad_values


class DinatLayer2d(_DinatLayerNd):
  def __init__(
    self,
    cfg: Config,
    dim: int,
    num_heads: int,
    kernel_size: int,
    dilation: int,
    drop_path_rate: float
  ):
    super().__init__(cfg, dim, kernel_size, dilation, drop_path_rate, double_attention=False)
    self.attention = NeighborhoodAttentionModule2d(cfg, dim, num_heads, kernel_size, dilation)
    self.attention2 = None

  def maybe_pad(self, hidden_states, height, width):
    window_size = self.window_size
    pad_values = (0, 0, 0, 0, 0, 0)
    if height < window_size or width < window_size:
      pad_l = pad_t = 0
      pad_r = max(0, window_size - width)
      pad_b = max(0, window_size - height)
      pad_values = (0, 0, pad_l, pad_r, pad_t, pad_b)
      hidden_states = mx.pad(hidden_states, ((0, 0), (pad_t, pad_b), (pad_l, pad_r), (0, 0)))
    return hidden_states, pad_values
