import json
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from numpy.typing import NDArray

PathLike = Union[str, PathLike]


@dataclass
class AllInOneOutput:
  logits_beat: Any = None
  logits_downbeat: Any = None
  logits_section: Any = None
  logits_function: Any = None
  embeddings: Any = None


@dataclass
class Segment:
  start: float
  end: float
  label: str


@dataclass
class AnalysisResult:
  path: Path
  bpm: int
  beats: List[float]
  downbeats: List[float]
  beat_positions: List[int]
  segments: List[Segment]
  activations: Optional[Dict[str, NDArray]] = None
  embeddings: Optional[NDArray] = None

  @staticmethod
  def from_json(
    path: PathLike,
    load_activations: bool = True,
    load_embeddings: bool = True,
  ):
    from .utils import mkpath

    path = mkpath(path)
    with open(path, 'r') as f:
      data = json.load(f)

    result = AnalysisResult(
      path=mkpath(data['path']),
      bpm=data['bpm'],
      beats=data['beats'],
      downbeats=data['downbeats'],
      beat_positions=data['beat_positions'],
      segments=[Segment(**seg) for seg in data['segments']],
    )

    if load_activations:
      activ_path = path.with_suffix('.activ.npz')
      if activ_path.is_file():
        activs = np.load(activ_path)
        result.activations = {key: activs[key] for key in activs.files}

    if load_embeddings:
      embed_path = path.with_suffix('.embed.npy')
      if embed_path.is_file():
        result.embeddings = np.load(embed_path)

    return result
