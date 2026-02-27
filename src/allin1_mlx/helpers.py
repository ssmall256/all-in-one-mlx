import json
import os
import time
from dataclasses import asdict
from glob import glob
from pathlib import Path
from typing import List, Union

import numpy as np

from .postprocessing import estimate_tempo_from_beats
from .postprocessing.functional_mlx import postprocess_functional_structure_mlx
from .postprocessing.metrical_mlx import postprocess_metrical_structure_mlx
from .typings import AllInOneOutput, AnalysisResult, PathLike
from .utils import compact_json_number_array, mkpath


def run_inference_mlx(
  path: Path,
  spec_path: Path,
  model,
  include_activations: bool,
  include_embeddings: bool,
  compile_forward: bool = False,
  timings: dict = None,
) -> AnalysisResult:
  import mlx.core as mx

  t0 = time.perf_counter()
  spec = np.load(spec_path, mmap_mode="r")
  spec = mx.array(spec)[None, ...]

  t1 = time.perf_counter()
  return _run_inference_mlx_spec(
    path=path,
    spec=spec,
    model=model,
    include_activations=include_activations,
    include_embeddings=include_embeddings,
    compile_forward=compile_forward,
    timings=timings,
    spec_timing=(t0, t1),
  )


def run_inference_mlx_spec(
  path: Path,
  spec: np.ndarray,
  model,
  include_activations: bool,
  include_embeddings: bool,
  compile_forward: bool = False,
  timings: dict = None,
) -> AnalysisResult:
  import mlx.core as mx

  def _is_mx_array(value) -> bool:
    return value is not None and value.__class__.__module__.startswith("mlx")

  spec_mx = spec if _is_mx_array(spec) else mx.array(spec)

  return _run_inference_mlx_spec(
    path=path,
    spec=spec_mx[None, ...],
    model=model,
    include_activations=include_activations,
    include_embeddings=include_embeddings,
    compile_forward=compile_forward,
    timings=timings,
  )


def _run_inference_mlx_spec(
  path: Path,
  spec,
  model,
  include_activations: bool,
  include_embeddings: bool,
  compile_forward: bool,
  timings: dict = None,
  spec_timing: tuple | None = None,
) -> AnalysisResult:
  import mlx.core as mx

  if compile_forward and os.environ.get("NATTEN_MLX_BACKEND", "").strip().lower() == "mlx":
    compile_forward = False

  t1 = time.perf_counter()
  forward = model
  if compile_forward:
    compiled_attr = "_compiled_forward_with_embeddings" if include_embeddings else "_compiled_forward_no_embeddings"
    if not hasattr(model, compiled_attr):
      from functools import partial
      state = [model.state]
      if include_embeddings:
        @partial(mx.compile, inputs=state)
        def _forward(x):
          outputs = model(x, return_embeddings=True)
          return (
            outputs.logits_beat,
            outputs.logits_downbeat,
            outputs.logits_section,
            outputs.logits_function,
            outputs.embeddings,
          )
      else:
        @partial(mx.compile, inputs=state)
        def _forward(x):
          outputs = model(x, return_embeddings=False)
          return (
            outputs.logits_beat,
            outputs.logits_downbeat,
            outputs.logits_section,
            outputs.logits_function,
          )
      setattr(model, compiled_attr, _forward)
    forward = getattr(model, compiled_attr)
    if include_embeddings:
      logits_beat, logits_downbeat, logits_section, logits_function, embeddings = forward(spec)
    else:
      logits_beat, logits_downbeat, logits_section, logits_function = forward(spec)
      embeddings = None
    logits = AllInOneOutput(
      logits_beat=logits_beat,
      logits_downbeat=logits_downbeat,
      logits_section=logits_section,
      logits_function=logits_function,
      embeddings=embeddings,
    )
  else:
    logits = forward(spec, return_embeddings=include_embeddings)

  # Compute probabilities from logits (cheap ops that extend the lazy graph).
  prob_beat_mx = mx.sigmoid(logits.logits_beat[0])
  prob_downbeat_mx = mx.sigmoid(logits.logits_downbeat[0])
  prob_section_mx = mx.sigmoid(logits.logits_section[0])
  prob_function_mx = mx.softmax(logits.logits_function[0], axis=0)

  # Single eval: materialises the forward pass + probabilities together.
  to_eval = [prob_beat_mx, prob_downbeat_mx, prob_section_mx, prob_function_mx]
  if include_embeddings and logits.embeddings is not None:
    to_eval.append(logits.embeddings)
  mx.eval(*to_eval)
  t2 = time.perf_counter()
  prob_beat = np.array(prob_beat_mx, copy=False)
  prob_downbeat = np.array(prob_downbeat_mx, copy=False)
  prob_section = np.array(prob_section_mx, copy=False)
  prob_function = np.array(prob_function_mx, copy=False)

  metrical_timings = {}
  functional_timings = {}
  metrical_structure = postprocess_metrical_structure_mlx(
    logits,
    model.cfg,
    prob_beat=prob_beat,
    prob_downbeat=prob_downbeat,
    timings=metrical_timings,
  )
  functional_structure = postprocess_functional_structure_mlx(
    logits,
    model.cfg,
    prob_sections=prob_section,
    prob_functions=prob_function,
    timings=functional_timings,
  )
  t3 = time.perf_counter()
  if timings is not None:
    if spec_timing is not None:
      timings["spec_load"] = spec_timing
    timings["nn"] = (t1, t2)
    timings["postprocess"] = (t2, t3)
    for key, value in metrical_timings.items():
      timings[key] = value
    for key, value in functional_timings.items():
      timings[key] = value
  bpm = estimate_tempo_from_beats(metrical_structure['beats'])

  result = AnalysisResult(
    path=path,
    bpm=bpm,
    segments=functional_structure,
    **metrical_structure,
  )

  if include_activations or include_embeddings:
    to_eval = []
    if include_activations:
      activations = {
        'beat': prob_beat,
        'downbeat': prob_downbeat,
        'segment': prob_section,
        'label': prob_function,
      }
    if include_embeddings:
      embeddings = logits.embeddings[0]
      to_eval.append(embeddings)
    if to_eval:
      mx.eval(*to_eval)

  if include_activations:
    result.activations = {
      'beat': np.array(activations['beat'], copy=False),
      'downbeat': np.array(activations['downbeat'], copy=False),
      'segment': np.array(activations['segment'], copy=False),
      'label': np.array(activations['label'], copy=False),
    }

  if include_embeddings:
    result.embeddings = np.array(embeddings, copy=False)

  return result


def run_inference_mlx_batch(
  paths: List[Path],
  specs: List[np.ndarray],
  model,
  include_activations: bool,
  include_embeddings: bool,
  compile_forward: bool = False,
  spec_timings: List[tuple] = None,
  timings_list: List[dict] = None,
) -> List[AnalysisResult]:
  import mlx.core as mx

  if timings_list is None:
    timings_list = [{} for _ in paths]

  if spec_timings is not None:
    for timings, spec_timing in zip(timings_list, spec_timings):
      timings["spec_load"] = spec_timing

  spec_batch = np.stack(specs, axis=0)
  t1 = time.perf_counter()
  forward = model
  if compile_forward:
    compiled_attr = "_compiled_forward_with_embeddings" if include_embeddings else "_compiled_forward_no_embeddings"
    if not hasattr(model, compiled_attr):
      from functools import partial
      state = [model.state]
      if include_embeddings:
        @partial(mx.compile, inputs=state)
        def _forward(x):
          outputs = model(x, return_embeddings=True)
          return (
            outputs.logits_beat,
            outputs.logits_downbeat,
            outputs.logits_section,
            outputs.logits_function,
            outputs.embeddings,
          )
      else:
        @partial(mx.compile, inputs=state)
        def _forward(x):
          outputs = model(x, return_embeddings=False)
          return (
            outputs.logits_beat,
            outputs.logits_downbeat,
            outputs.logits_section,
            outputs.logits_function,
          )
      setattr(model, compiled_attr, _forward)
    forward = getattr(model, compiled_attr)
    if include_embeddings:
      logits_beat, logits_downbeat, logits_section, logits_function, embeddings = forward(mx.array(spec_batch))
      mx.eval(logits_beat, logits_downbeat, logits_section, logits_function, embeddings)
    else:
      logits_beat, logits_downbeat, logits_section, logits_function = forward(mx.array(spec_batch))
      mx.eval(logits_beat, logits_downbeat, logits_section, logits_function)
      embeddings = None
    logits = AllInOneOutput(
      logits_beat=logits_beat,
      logits_downbeat=logits_downbeat,
      logits_section=logits_section,
      logits_function=logits_function,
      embeddings=embeddings,
    )
  else:
    logits = forward(mx.array(spec_batch), return_embeddings=include_embeddings)
    to_eval = [
      logits.logits_beat,
      logits.logits_downbeat,
      logits.logits_section,
      logits.logits_function,
    ]
    if include_embeddings and logits.embeddings is not None:
      to_eval.append(logits.embeddings)
    mx.eval(*to_eval)
  t2 = time.perf_counter()

  prob_beat_mx = mx.sigmoid(logits.logits_beat)
  prob_downbeat_mx = mx.sigmoid(logits.logits_downbeat)
  prob_section_mx = mx.sigmoid(logits.logits_section)
  prob_function_mx = mx.softmax(logits.logits_function, axis=1)
  mx.eval(prob_beat_mx, prob_downbeat_mx, prob_section_mx, prob_function_mx)
  prob_beat = np.array(prob_beat_mx, copy=False)
  prob_downbeat = np.array(prob_downbeat_mx, copy=False)
  prob_section = np.array(prob_section_mx, copy=False)
  prob_function = np.array(prob_function_mx, copy=False)

  for timings in timings_list:
    timings["nn"] = (t1, t2)

  results = []
  for idx, path in enumerate(paths):
    t_post_start = time.perf_counter()
    item_logits = AllInOneOutput(
      logits_beat=logits.logits_beat[idx:idx + 1],
      logits_downbeat=logits.logits_downbeat[idx:idx + 1],
      logits_section=logits.logits_section[idx:idx + 1],
      logits_function=logits.logits_function[idx:idx + 1],
      embeddings=(logits.embeddings[idx:idx + 1] if logits.embeddings is not None else None),
    )

    metrical_timings = {}
    functional_timings = {}
    metrical_structure = postprocess_metrical_structure_mlx(
      item_logits,
      model.cfg,
      prob_beat=prob_beat[idx],
      prob_downbeat=prob_downbeat[idx],
      timings=metrical_timings,
    )
    functional_structure = postprocess_functional_structure_mlx(
      item_logits,
      model.cfg,
      prob_sections=prob_section[idx],
      prob_functions=prob_function[idx],
      timings=functional_timings,
    )
    t_post_end = time.perf_counter()

    if timings_list is not None:
      timings = timings_list[idx]
      timings["postprocess"] = (t_post_start, t_post_end)
      for key, value in metrical_timings.items():
        timings[key] = value
      for key, value in functional_timings.items():
        timings[key] = value

    bpm = estimate_tempo_from_beats(metrical_structure['beats'])
    result = AnalysisResult(
      path=path,
      bpm=bpm,
      segments=functional_structure,
      **metrical_structure,
    )

    if include_activations or include_embeddings:
      to_eval = []
      if include_activations:
        activations = {
          'beat': prob_beat[idx],
          'downbeat': prob_downbeat[idx],
          'segment': prob_section[idx],
          'label': prob_function[idx],
        }
      if include_embeddings:
        embeddings = item_logits.embeddings[0]
        to_eval.append(embeddings)
      if to_eval:
        mx.eval(*to_eval)

    if include_activations:
      result.activations = {
        'beat': np.array(activations['beat'], copy=False),
        'downbeat': np.array(activations['downbeat'], copy=False),
        'segment': np.array(activations['segment'], copy=False),
        'label': np.array(activations['label'], copy=False),
      }

    if include_embeddings:
      result.embeddings = np.array(embeddings, copy=False)

    results.append(result)

  return results


def expand_paths(paths: List[Path]):
  expanded_paths = set()
  for path in paths:
    if '*' in str(path) or '?' in str(path):
      matches = [Path(p) for p in glob(str(path))]
      if not matches:
        raise FileNotFoundError(f'Could not find any files matching {path}')
      expanded_paths.update(matches)
    else:
      expanded_paths.add(path)

  return sorted(expanded_paths)


def check_paths(paths: List[Path]):
  missing_files = []
  for path in paths:
    if not path.is_file():
      missing_files.append(str(path))
  if missing_files:
    raise FileNotFoundError(f'Could not find the following files: {missing_files}')


def rmdir_if_empty(path: Path):
  try:
    path.rmdir()
  except (FileNotFoundError, OSError):
    pass


def _round_floats(data, decimals=2):
  """Round float values in nested data structures to specified decimal places."""
  if isinstance(data, float):
    return round(data, decimals)
  elif isinstance(data, dict):
    return {key: _round_floats(value, decimals) for key, value in data.items()}
  elif isinstance(data, list):
    return [_round_floats(item, decimals) for item in data]
  else:
    return data


def save_results(
  results: Union[AnalysisResult, List[AnalysisResult]],
  out_dir: PathLike,
):
  if not isinstance(results, list):
    results = [results]

  out_dir = mkpath(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  for result in results:
    out_path = out_dir / result.path.with_suffix('.json').name
    result = asdict(result)
    result['path'] = str(result['path'])

    activations = result.pop('activations')
    if activations is not None:
      np.savez(str(out_path.with_suffix('.activ.npz')), **activations)

    embeddings = result.pop('embeddings')
    if embeddings is not None:
      np.save(str(out_path.with_suffix('.embed.npy')), embeddings)

    # Round all float values to 2 decimal places
    result = _round_floats(result, decimals=2)

    json_str = json.dumps(result, indent=2)
    json_str = compact_json_number_array(json_str)
    out_path.with_suffix('.json').write_text(json_str)
