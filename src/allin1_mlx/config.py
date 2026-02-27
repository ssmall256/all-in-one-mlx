from dataclasses import dataclass, field
from typing import Any, List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

HARMONIX_LABELS = [
  'start',
  'end',
  'intro',
  'outro',
  'break',
  'bridge',
  'inst',
  'solo',
  'verse',
  'chorus',
]


@dataclass
class DataConfig:
  name: str
  num_instruments: int
  num_labels: int


defaults = [
  '_self_',
  {'data': 'harmonix'},
]


@dataclass
class Config:
  model: str = 'allinone'

  data: DataConfig = MISSING
  defaults: List[Any] = field(default_factory=lambda: defaults)

  # Audio configurations ---------------------------------------------------
  sample_rate: int = 44100
  window_size: int = 2048
  num_bands: int = 12
  hop_size: int = 441  # FPS=100
  fps: int = 100
  fmin: int = 30
  fmax: int = 17000

  # Model configurations ---------------------------------------------------
  threshold_beat: float = 0.19
  threshold_downbeat: float = 0.19
  threshold_section: float = 0.05

  best_threshold_beat: Optional[float] = None
  best_threshold_downbeat: Optional[float] = None

  instrument_attention: bool = True
  double_attention: bool = True

  depth: int = 11
  dilation_factor: int = 2
  dilation_max: int = 3200
  num_heads: int = 2
  kernel_size: int = 5

  dim_input: int = 81
  dim_embed: int = 24
  mlp_ratio: float = 4.0
  qkv_bias: bool = True

  drop_conv: float = 0.2
  drop_path: float = 0.1
  drop_hidden: float = 0.2
  drop_attention: float = 0.2
  drop_last: float = 0.0

  act_conv: str = 'elu'
  act_transformer: str = 'gelu'

  layer_norm_eps: float = 1e-5

  # Misc -------------------------------------------------------------------
  seed: int = 1234
  fold: int = 2
  aafold: Optional[int] = None
  total_folds: int = 8

  bpm_min: int = 55
  bpm_max: int = 240
  min_hops_per_beat: int = 24  # 60 / max_bpm * sample_rate / hop_size


_harmonix_data = DataConfig(name='harmonix', num_instruments=4, num_labels=10)

cs = ConfigStore.instance()
cs.store(name='config', node=Config)
cs.store(group='data', name='harmonix', node=_harmonix_data)
cs.store(name='harmonix', node=_harmonix_data)
