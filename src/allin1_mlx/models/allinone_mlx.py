import functools
import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..config import Config
from ..typings import AllInOneOutput
from .utils_mlx import get_activation_function


@functools.lru_cache(maxsize=1)
def _load_dinat_layers():
  try:
    from .dinat_mlx import DinatLayer1d, DinatLayer2d
  except Exception as exc:
    raise ImportError(
      "MLX Dinat layers are required for the MLX AllInOne model. "
      "Provide allin1_mlx.models.dinat_mlx with DinatLayer1d/DinatLayer2d."
    ) from exc
  return DinatLayer1d, DinatLayer2d


class AllInOneMLX(nn.Module):
  def __init__(self, cfg: Config):
    super().__init__()

    self.cfg = cfg
    self.num_levels = cfg.depth
    self.num_features = int(cfg.dim_embed * 2 ** (self.num_levels - 1))

    self.embeddings = AllInOneEmbeddingsMLX(cfg)
    self.encoder = AllInOneEncoderMLX(cfg, depth=cfg.depth)

    self.norm = nn.LayerNorm(cfg.dim_embed, eps=cfg.layer_norm_eps)

    self.beat_classifier = HeadMLX(num_classes=1, cfg=cfg, init_confidence=0.05)
    self.downbeat_classifier = HeadMLX(num_classes=1, cfg=cfg, init_confidence=0.0125)
    self.section_classifier = HeadMLX(num_classes=1, cfg=cfg, init_confidence=0.001)
    self.function_classifier = HeadMLX(num_classes=cfg.data.num_labels, cfg=cfg)

    self.dropout = nn.Dropout(cfg.drop_last)

  def __call__(
    self,
    inputs: mx.array,
    output_attentions: Optional[bool] = None,
  ):
    # N: batch size
    # K: instrument
    # T: time
    # F: frequency
    # inputs has shape of: N, K, T, F
    n, k, t, f = inputs.shape

    inputs = mx.reshape(inputs, (n * k, t, f, 1))
    frame_embed = self.embeddings(inputs)  # NK, T, C

    encoder_outputs = self.encoder(
      frame_embed,
      output_attentions=output_attentions,
    )
    hidden_state_levels = encoder_outputs[0]

    hidden_states = mx.reshape(hidden_state_levels[-1], (n, k, t, -1))
    hidden_states = self.norm(hidden_states)
    hidden_states = self.dropout(hidden_states)

    # Optimization: Transpose and reshape once for all heads
    # Shape: (N, K, T, C) -> (N, T, K, C) -> (N, T, K*C)
    batch, inst, frame, embed = hidden_states.shape
    hidden_states_transposed = mx.transpose(hidden_states, (0, 2, 1, 3))
    hidden_states_flat = mx.reshape(hidden_states_transposed, (batch, frame, inst * embed))

    logits_beat = self.beat_classifier(hidden_states_flat, preprocessed=True)
    logits_downbeat = self.downbeat_classifier(hidden_states_flat, preprocessed=True)
    logits_section = self.section_classifier(hidden_states_flat, preprocessed=True)
    logits_function = self.function_classifier(hidden_states_flat, preprocessed=True)

    return AllInOneOutput(
      logits_beat=logits_beat,
      logits_downbeat=logits_downbeat,
      logits_section=logits_section,
      logits_function=logits_function,
      embeddings=hidden_states,
    )


class AllInOneEncoderMLX(nn.Module):
  def __init__(self, cfg: Config, depth: int):
    super().__init__()
    self.cfg = cfg

    drop_path_rates = np.linspace(0.0, cfg.drop_path, depth).tolist()
    dilations = [
      min(cfg.dilation_factor ** i, cfg.dilation_max)
      for i in range(depth)
    ]
    self.layers = [
      AllInOneBlockMLX(
        cfg=cfg,
        dilation=dilations[i],
        drop_path_rate=drop_path_rates[i],
      )
      for i in range(depth)
    ]

  def __call__(
    self,
    frame_embed: mx.array,
    output_attentions: Optional[bool] = None,
  ):
    # frame_embed has shape of: NK, T, C
    # Optimization: Only store last layer to save memory
    all_attentions = [] if output_attentions else None
    hidden_states = frame_embed
    for layer in self.layers:
      layer_outputs = layer(hidden_states, output_attentions)
      hidden_states = layer_outputs[0]
      if output_attentions:
        all_attentions.append(layer_outputs[1:])

    # Return last hidden state wrapped in list for compatibility
    outputs = ([hidden_states],)
    if output_attentions:
      outputs += (all_attentions,)
    return outputs


class AllInOneBlockMLX(nn.Module):
  def __init__(self, cfg: Config, dilation: int, drop_path_rate: float):
    super().__init__()

    self.cfg = cfg
    self.dilation = dilation
    dinat_1d, dinat_2d = _load_dinat_layers()

    self.timelayer = dinat_1d(
      cfg=cfg,
      dim=cfg.dim_embed,
      num_heads=cfg.num_heads,
      kernel_size=cfg.kernel_size,
      dilation=dilation,
      drop_path_rate=drop_path_rate,
      double_attention=cfg.double_attention,
    )

    if cfg.instrument_attention:
      self.instlayer = dinat_2d(
        cfg=cfg,
        dim=cfg.dim_embed,
        num_heads=cfg.num_heads,
        kernel_size=5,
        dilation=1,
        drop_path_rate=drop_path_rate,
      )
    else:
      self.instlayer = dinat_1d(
        cfg=cfg,
        dim=cfg.dim_embed,
        num_heads=cfg.num_heads,
        kernel_size=5,
        dilation=1,
        drop_path_rate=drop_path_rate,
        double_attention=False,
      )

  def __call__(
    self,
    hidden_states: mx.array,
    output_attentions: Optional[bool] = None,
  ):
    # hidden_states has shape of: NK, T, C
    nk, t, c = hidden_states.shape
    n, k = nk // self.cfg.data.num_instruments, self.cfg.data.num_instruments

    timelayer_outputs = self.timelayer(hidden_states, output_attentions)
    hidden_states = timelayer_outputs[0]
    if self.cfg.instrument_attention:
      hidden_states = mx.reshape(hidden_states, (n, k, t, c))
      instlayer_outputs = self.instlayer(hidden_states, output_attentions)
      hidden_states = instlayer_outputs[0]
      hidden_states = mx.reshape(hidden_states, (nk, t, c))
    else:
      instlayer_outputs = self.instlayer(hidden_states, output_attentions)
      hidden_states = instlayer_outputs[0]

    outputs = (hidden_states,)
    if output_attentions:
      outputs += timelayer_outputs[1:]
      if self.instlayer is not None:
        outputs += instlayer_outputs[1:]
    return outputs


class AllInOneEmbeddingsMLX(nn.Module):
  def __init__(self, cfg: Config):
    super().__init__()
    dim_input, hidden_size = cfg.dim_input, cfg.dim_embed
    self.dim_input = dim_input
    self.hidden_size = hidden_size

    self.act_fn = get_activation_function(cfg.act_conv)
    first_conv_filters = hidden_size if cfg.model == 'tcn' else hidden_size // 2

    self.conv0 = nn.Conv2d(1, first_conv_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 0))
    self.pool0 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3), padding=(0, 0))
    self.drop0 = nn.Dropout(cfg.drop_conv)

    self.conv1 = nn.Conv2d(first_conv_filters, hidden_size, kernel_size=(1, 12), stride=(1, 1), padding=(0, 0))
    self.pool1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3), padding=(0, 0))
    self.drop1 = nn.Dropout(cfg.drop_conv)

    self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=(3, 3), stride=(1, 1), padding=(1, 0))
    self.pool2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3), padding=(0, 0))

    self.norm = nn.LayerNorm(cfg.dim_embed)
    self.dropout = nn.Dropout(cfg.drop_conv)

  def __call__(self, x: mx.array):
    # x has shape of: NK, T, F, C=1
    x = self.conv0(x)
    x = self.pool0(x)
    x = self.act_fn(x)
    x = self.drop0(x)

    x = self.conv1(x)
    x = self.pool1(x)
    x = self.act_fn(x)
    x = self.drop1(x)

    x = self.conv2(x)
    x = self.pool2(x)
    x = self.act_fn(x)

    embeddings = mx.squeeze(x, axis=2)
    embeddings = self.norm(embeddings)
    embeddings = self.dropout(embeddings)

    return embeddings


class HeadMLX(nn.Module):
  def __init__(self, num_classes: int, cfg: Config, init_confidence: float = None):
    super().__init__()
    self.classifier = nn.Linear(cfg.data.num_instruments * cfg.dim_embed, num_classes)

    if init_confidence is not None:
      self.reset_parameters(init_confidence)

  def reset_parameters(self, confidence) -> None:
    value = -math.log(1 / confidence - 1)
    if hasattr(self.classifier, "bias"):
      self.classifier.bias = mx.full(self.classifier.bias.shape, value)

  def __call__(self, x: mx.array, preprocessed: bool = False):
    # If preprocessed=True, x shape is already: (N, T, K*C)
    # If preprocessed=False, x shape is: (N, K, T, C) and we do transformation
    if not preprocessed:
      # x shape: N, K, T, C
      batch, inst, frame, embed = x.shape
      x = mx.transpose(x, (0, 2, 1, 3))
      x = mx.reshape(x, (batch, frame, inst * embed))

    logits = self.classifier(x)
    logits = mx.transpose(logits, (0, 2, 1))
    if logits.shape[1] == 1:
      logits = mx.squeeze(logits, axis=1)
    return logits
