import copy
from typing import List

import mlx.core as mx
import mlx.nn as nn

from ..typings import AllInOneOutput


class EnsembleMLX(nn.Module):
  def __init__(self, models: List[nn.Module], parallel: bool = True):
    super().__init__()

    cfg = copy.deepcopy(models[0].cfg)
    cfg.best_threshold_beat = sum([model.cfg.best_threshold_beat for model in models]) / len(models)
    cfg.best_threshold_downbeat = sum([model.cfg.best_threshold_downbeat for model in models]) / len(models)

    self.cfg = cfg
    self.models = models
    # Note: parallel parameter accepted but not used - kept for API compatibility

  def __call__(self, x, return_embeddings=True):
    beat_logits = []
    downbeat_logits = []
    section_logits = []
    function_logits = []
    embeddings_list = []

    # Collect outputs from all models
    for model in self.models:
      output = model(x)
      beat_logits.append(output.logits_beat)
      downbeat_logits.append(output.logits_downbeat)
      section_logits.append(output.logits_section)
      function_logits.append(output.logits_function)
      if return_embeddings:
        embeddings_list.append(output.embeddings)

    # Stack all outputs (lazy operations)
    beat_stack = mx.stack(beat_logits, axis=0)
    downbeat_stack = mx.stack(downbeat_logits, axis=0)
    section_stack = mx.stack(section_logits, axis=0)
    function_stack = mx.stack(function_logits, axis=0)

    if return_embeddings:
      embeddings_stack = mx.stack(embeddings_list, axis=-1)
    else:
      embeddings_stack = None

    # Note: Don't call mx.eval here - it breaks when this function is compiled
    # Evaluation happens in the caller (helpers.py)

    # Compute means (reuse stacks to avoid recreation)
    avg = AllInOneOutput(
      logits_beat=mx.mean(beat_stack, axis=0),
      logits_downbeat=mx.mean(downbeat_stack, axis=0),
      logits_section=mx.mean(section_stack, axis=0),
      logits_function=mx.mean(function_stack, axis=0),
      embeddings=embeddings_stack,
    )

    return avg
