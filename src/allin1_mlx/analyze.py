import json
import os
import queue
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
from tqdm import tqdm

from .demix import demix
from .helpers import (
  check_paths,
  expand_paths,
  rmdir_if_empty,
  run_inference_mlx,
  run_inference_mlx_batch,
  run_inference_mlx_spec,
  save_results,
)
from .models import load_pretrained_model_mlx
from .spectrogram import extract_spectrograms, spectrogram_from_stems
from .typings import AnalysisResult, PathLike
from .utils import load_result, mkpath


def _ensure_mlx_env(device: str) -> None:
  if device != "mlx":
    return
  os.environ.setdefault("NATTEN_MLX", "1")
  os.environ.setdefault("NATTEN_MLX_BACKEND", "metal")
  os.environ.setdefault("NATTEN_MLX_COMPILE", "1")


def _run_mlx_inference(
  todo_paths: List[Path],
  spec_paths_for_infer: List[Path],
  model_container: Dict[str, Any],
  model_loader_thread: Optional[threading.Thread],
  model_name: str,
  mlx_weights_dir: Optional[Path],
  mlx_weights_path: Optional[Path],
  mlx_config_path: Optional[Path],
  include_activations: bool,
  include_embeddings: bool,
  timings_embed: bool,
  mlx_batch_size: int,
  mlx_compile: bool,
  ensemble_parallel: bool,
  out_dir: Optional[Path],
  _emit_timing,
  _emit_summary,
) -> List[AnalysisResult]:
  """Run MLX-based inference on audio tracks."""
  results = []

  if model_loader_thread is not None:
    model_loader_thread.join()
    if model_container['error'] is not None:
      raise model_container['error']
    if model_container['model'] is not None:
      model = model_container['model']
      t0, t1 = model_container['load_time']
      _emit_timing("model_load", None, t0, t1)
    else:
      t0 = time.perf_counter()
      model = load_pretrained_model_mlx(
        model_name=model_name,
        weights_dir=mlx_weights_dir,
        weights_path=mlx_weights_path,
        config_path=mlx_config_path,
        ensemble_parallel=ensemble_parallel,
      )
      t1 = time.perf_counter()
      _emit_timing("model_load", None, t0, t1)
  else:
    t0 = time.perf_counter()
    model = load_pretrained_model_mlx(
      model_name=model_name,
      weights_dir=mlx_weights_dir,
      weights_path=mlx_weights_path,
      config_path=mlx_config_path,
    )
    t1 = time.perf_counter()
    _emit_timing("model_load", None, t0, t1)

  batch_size = max(1, int(mlx_batch_size))
  if batch_size <= 1:
    pbar = tqdm(zip(todo_paths, spec_paths_for_infer), total=len(todo_paths))
    for path, spec_path in pbar:
      pbar.set_description(f'Analyzing {path.name}')

      timings = {}
      result = run_inference_mlx(
        path=path,
        spec_path=spec_path,
        model=model,
        include_activations=include_activations,
        include_embeddings=include_embeddings,
        compile_forward=mlx_compile,
        timings=timings,
      )
      if "nn" in timings:
        _emit_timing("nn", path, *timings["nn"])
      if "spec_load" in timings:
        _emit_timing("spec_load", path, *timings["spec_load"])
      if "postprocess" in timings:
        _emit_timing("postprocess", path, *timings["postprocess"])
      for key in (
        "metrical_prep",
        "metrical_dbn",
        "functional_probs",
        "functional_local_maxima",
        "functional_boundaries",
      ):
        if key in timings:
          _emit_timing(key, path, *timings[key])
      if timings_embed:
        result.activations = result.activations or {}
        result.activations["timings"] = {
          "nn": (timings["nn"][1] - timings["nn"][0]) if "nn" in timings else None,
          "postprocess": (timings["postprocess"][1] - timings["postprocess"][0]) if "postprocess" in timings else None,
        }

      if _emit_summary is not None:
        _emit_summary(path, timings)

      if out_dir is not None:
        t0 = time.perf_counter()
        save_results(result, out_dir)
        t1 = time.perf_counter()
        _emit_timing("save", path, t0, t1)

      results.append(result)
  else:
    pbar = tqdm(zip(todo_paths, spec_paths_for_infer), total=len(todo_paths))
    batch_paths = []
    batch_specs = []
    batch_spec_timings = []
    batch_shape = None

    def _flush_batch():
      nonlocal batch_paths, batch_specs, batch_spec_timings, batch_shape
      if not batch_paths:
        return
      timings_list = [{} for _ in batch_paths]
      batch_results = run_inference_mlx_batch(
        paths=batch_paths,
        specs=batch_specs,
        model=model,
        include_activations=include_activations,
        include_embeddings=include_embeddings,
        compile_forward=mlx_compile,
        spec_timings=batch_spec_timings,
        timings_list=timings_list,
      )
      for result, timings in zip(batch_results, timings_list):
        path = result.path
        if "nn" in timings:
          _emit_timing("nn", path, *timings["nn"])
        if "spec_load" in timings:
          _emit_timing("spec_load", path, *timings["spec_load"])
        if "postprocess" in timings:
          _emit_timing("postprocess", path, *timings["postprocess"])
        for key in (
          "metrical_prep",
          "metrical_dbn",
          "functional_probs",
          "functional_local_maxima",
          "functional_boundaries",
        ):
          if key in timings:
            _emit_timing(key, path, *timings[key])
        if timings_embed:
          result.activations = result.activations or {}
          result.activations["timings"] = {
            "nn": (timings["nn"][1] - timings["nn"][0]) if "nn" in timings else None,
            "postprocess": (timings["postprocess"][1] - timings["postprocess"][0]) if "postprocess" in timings else None,
          }
        if _emit_summary is not None:
          _emit_summary(path, timings)
        if out_dir is not None:
          t0 = time.perf_counter()
          save_results(result, out_dir)
          t1 = time.perf_counter()
          _emit_timing("save", path, t0, t1)
        results.append(result)
      batch_paths = []
      batch_specs = []
      batch_spec_timings = []
      batch_shape = None

    for path, spec_path in pbar:
      pbar.set_description(f'Analyzing {path.name}')
      t0 = time.perf_counter()
      spec = np.load(spec_path, mmap_mode="r")
      spec = np.array(spec)
      t1 = time.perf_counter()
      spec_shape = spec.shape
      if batch_shape is None:
        batch_shape = spec_shape
      if spec_shape != batch_shape or len(batch_paths) >= batch_size:
        _flush_batch()
        batch_shape = spec_shape
      batch_paths.append(path)
      batch_specs.append(spec)
      batch_spec_timings.append((t0, t1))
    _flush_batch()

  return results


def _parse_overwrite(overwrite: Union[bool, str, None]) -> Set[str]:
  if overwrite is True:
    return {"demix", "spec", "json", "viz", "sonify"}
  if not overwrite:
    return set()
  if isinstance(overwrite, str):
    value = overwrite.strip().lower()
    if value == "all":
      return {"demix", "spec", "json", "viz", "sonify"}
    stages = {part.strip() for part in value.split(",") if part.strip()}
    valid = {"demix", "spec", "json", "viz", "sonify"}
    unknown = stages - valid
    if unknown:
      raise ValueError(f"Unknown overwrite stage(s): {sorted(unknown)}")
    return stages
  return set()


def analyze(
  paths: Union[PathLike, List[PathLike]],
  out_dir: PathLike = None,
  visualize: Union[bool, PathLike] = False,
  sonify: Union[bool, PathLike] = False,
  model: str = 'harmonix-all',
  device: str = 'mlx',
  include_activations: bool = False,
  include_embeddings: bool = False,
  demix_dir: PathLike = './demix',
  spec_dir: PathLike = './spec',
  spec_backend: Optional[str] = None,
  keep_byproducts: bool = False,
  overwrite: Union[bool, str, None] = None,
  multiprocess: bool = True,
  mlx_weights_dir: PathLike = None,
  mlx_weights_path: PathLike = None,
  mlx_config_path: PathLike = None,
  mlx_batch_size: int = 1,
  mlx_compile: Optional[bool] = None,
  mlx_in_memory: Optional[bool] = None,
  ensemble_parallel: bool = True,
  spec_check: bool = False,
  timings_path: PathLike = None,
  timings_embed: bool = False,
  timings_viz_path: PathLike = None,
  timings_summary: bool = False,
  timings_summary_path: PathLike = None,
  timings_json_summary_path: PathLike = None,
  dbn_backend: Optional[str] = None,
) -> Union[AnalysisResult, List[AnalysisResult]]:
  """
  Analyzes the provided audio files and returns the analysis results.
  """

  model_name = model
  if device != "mlx":
    raise ValueError("This MLX-only build supports only device='mlx'.")
  if spec_backend is None:
    spec_backend = "mlx_fast"
  if spec_backend not in {"mlx", "mlx_fast"}:
    raise ValueError("spec_backend must be 'mlx' or 'mlx_fast' for MLX-only builds.")
  if dbn_backend is not None:
    dbn_backend = str(dbn_backend).strip().lower()
    if dbn_backend not in {"auto", "cpp", "cython", "numba", "python"}:
      raise ValueError("dbn_backend must be one of: auto, cpp, cython, numba, python.")
    os.environ["ALLIN1_DBN_BACKEND"] = dbn_backend
  else:
    os.environ.setdefault("ALLIN1_DBN_BACKEND", "auto")
  if mlx_in_memory is None:
    mlx_in_memory = True
  overwrite_set = _parse_overwrite(overwrite)
  overwrite_demix = "demix" in overwrite_set
  overwrite_spec = "spec" in overwrite_set
  overwrite_json = "json" in overwrite_set
  overwrite_viz = "viz" in overwrite_set
  overwrite_sonify = "sonify" in overwrite_set
  demix_stage = "demix: demucs-mlx"
  return_list = True
  if not isinstance(paths, list):
    return_list = False
    paths = [paths]
  if not paths:
    raise ValueError('At least one path must be specified.')
  paths = [mkpath(p) for p in paths]
  paths = expand_paths(paths)
  check_paths(paths)
  if mlx_compile is None:
    # For one-off single-track runs, compile overhead often dominates.
    mlx_compile = len(paths) > 1
  cache_stats_before = None  # Model cache stats not implemented

  temp_dir = None
  if not keep_byproducts:
    temp_dir = tempfile.mkdtemp(prefix='allin1_mlx_')
    demix_dir_actual = Path(temp_dir) / 'demix'
    spec_dir_actual = Path(temp_dir) / 'spec'
    demix_dir_actual.mkdir(parents=True, exist_ok=True)
    spec_dir_actual.mkdir(parents=True, exist_ok=True)
  else:
    demix_dir_actual = mkpath(demix_dir)
    spec_dir_actual = mkpath(spec_dir)

  _ensure_mlx_env(device)
  analyze_start = time.perf_counter()
  timings_path = Path(timings_path) if timings_path is not None else None
  timings_json_summary_path = (
    Path(timings_json_summary_path) if timings_json_summary_path is not None else None
  )
  timings_handle = None
  if timings_path is not None:
    timings_path.parent.mkdir(parents=True, exist_ok=True)
    timings_handle = timings_path.open('a')
  timing_records = [] if timings_json_summary_path is not None else None
  writer_thread = None
  write_queue = None
  timings_viz_pending = False
  summary_handle = None

  def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
      return value
    if isinstance(value, Path):
      return str(value)
    try:
      return float(value)
    except Exception:
      return str(value)

  def _emit_timing(stage: str, track: Optional[Path], start: float, end: float, extra: Optional[Dict[str, Any]] = None):
    if timings_handle is None and timing_records is None:
      return
    record: Dict[str, Any] = {
      "stage": stage,
      "start": round(float(start), 2),
      "end": round(float(end), 2),
      "duration": round(float(end - start), 2),
      "device": _json_safe(device),
      "model": _json_safe(model_name),
    }
    if track is not None:
      record["track"] = str(track)
    if extra:
      record.update({k: _json_safe(v) for k, v in extra.items()})
    if timings_handle is not None:
      timings_handle.write(json.dumps(record) + "\n")
      timings_handle.flush()
    if timing_records is not None:
      timing_records.append(record)

  def _emit_summary(track: Path, timings: Dict[str, Any]) -> None:
    if not timings_summary:
      return
    def _dur_value(key: str) -> Optional[float]:
      if key not in timings:
        return None
      start, end = timings[key]
      return float(end - start)
    def _dur(key: str) -> str:
      value = _dur_value(key)
      if value is None:
        return "na"
      return f"{value:.3f}s"
    print(
      f"=> timing_summary {track.name} "
      f"nn={_dur('nn')} postprocess={_dur('postprocess')} metrical_dbn={_dur('metrical_dbn')}"
    )
    if summary_handle is not None:
      nn_s = _dur_value("nn")
      post_s = _dur_value("postprocess")
      dbn_s = _dur_value("metrical_dbn")
      summary_handle.write(
        f"{track}\t{nn_s if nn_s is not None else 'na'}\t"
        f"{post_s if post_s is not None else 'na'}\t"
        f"{dbn_s if dbn_s is not None else 'na'}\n"
      )
      summary_handle.flush()

  try:
    if timings_summary_path is not None:
      timings_summary_path = Path(timings_summary_path)
      timings_summary_path.parent.mkdir(parents=True, exist_ok=True)
      summary_handle = timings_summary_path.open("a")
    if out_dir is None or overwrite_json:
      todo_paths = paths
      exist_paths = []
    else:
      out_paths = [mkpath(out_dir) / path.with_suffix('.json').name for path in paths]
      todo_paths = [path for path, out_path in zip(paths, out_paths) if not out_path.exists()]
      exist_paths = [out_path for path, out_path in zip(paths, out_paths) if out_path.exists()]
    process_paths = paths if (overwrite_demix or overwrite_spec) else todo_paths

    print(f'=> Found {len(exist_paths)} tracks already analyzed and {len(todo_paths)} tracks to analyze.')
    if exist_paths:
      print('=> To re-analyze, please use --overwrite option.')

    results = []
    if exist_paths:
      results += [
        load_result(
          exist_path,
          load_activations=include_activations,
          load_embeddings=include_embeddings,
        )
        for exist_path in tqdm(exist_paths, desc='Loading existing results')
      ]

    demix_paths = []
    spec_paths = []
    spec_map = {}

    model_loader_thread = None
    model_container = {'model': None, 'error': None, 'load_time': None}

    if todo_paths:
      def load_model_background():
        try:
          t_load_start = time.perf_counter()
          model_container['model'] = load_pretrained_model_mlx(
            model_name=model,
            weights_dir=mlx_weights_dir,
            weights_path=mlx_weights_path,
            config_path=mlx_config_path,
            ensemble_parallel=ensemble_parallel,
          )
          t_load_end = time.perf_counter()
          model_container['load_time'] = (t_load_start, t_load_end)
        except Exception as e:
          model_container['error'] = e

      model_loader_thread = threading.Thread(target=load_model_background, daemon=True)
      model_loader_thread.start()

    use_mlx_in_memory = (
      mlx_in_memory and
      device == "mlx" and
      spec_backend in ("mlx", "mlx_fast")
    )

    if process_paths and not use_mlx_in_memory:
      t0 = time.perf_counter()
      demix_paths = demix(process_paths, demix_dir_actual, overwrite=overwrite_demix)
      t1 = time.perf_counter()
      _emit_timing(demix_stage, None, t0, t1, {"count": len(process_paths)})

      t0 = time.perf_counter()
      spec_paths = extract_spectrograms(
        demix_paths,
        spec_dir_actual,
        multiprocess,
        overwrite=overwrite_spec,
        backend=spec_backend,
        check=spec_check,
      )
      t1 = time.perf_counter()
      _emit_timing("spectrogram", None, t0, t1, {"count": len(spec_paths)})
      spec_map = {demix_path.name: spec_path for demix_path, spec_path in zip(demix_paths, spec_paths)}

    if todo_paths:
      if use_mlx_in_memory:
        try:
          from demucs_mlx.api import save_audio
        except Exception as exc:
          raise ImportError(
            "demucs-mlx is not available. Install it with `uv pip install demucs-mlx`."
          ) from exc
        t_init0 = time.perf_counter()
        from demucs_mlx.api import Separator
        separator = Separator(model="htdemucs", progress=False)
        t_init1 = time.perf_counter()
        _emit_timing("demix_init: demucs-mlx", None, t_init0, t_init1)
        if keep_byproducts:
          write_queue = queue.Queue()

          def _writer():
            while True:
              item = write_queue.get()
              if item is None:
                break
              stems, out_dir, sr, spec, spec_path = item
              out_dir.mkdir(parents=True, exist_ok=True)
              for stem, audio in stems.items():
                save_audio(audio, out_dir / f"{stem}.wav", sr)
              spec_path.parent.mkdir(parents=True, exist_ok=True)
              np.save(str(spec_path), spec)

          writer_thread = threading.Thread(target=_writer, daemon=True)
          writer_thread.start()

        model = None

        pbar = tqdm(todo_paths, total=len(todo_paths))
        for path in pbar:
          pbar.set_description(f'Analyzing {path.name}')

          t0 = time.perf_counter()
          _, stems = separator.separate_audio_file(path, return_mx=True)
          t1 = time.perf_counter()
          _emit_timing(demix_stage, path, t0, t1)

          t2 = time.perf_counter()
          spec = spectrogram_from_stems(
            stems,
            sample_rate=separator.samplerate,
            backend=spec_backend,
            check=spec_check,
            return_mx=True,
          )
          t3 = time.perf_counter()
          _emit_timing("spectrogram", path, t2, t3)

          # Deferred join: model loads in background while first track demixes.
          if model is None:
            model_loader_thread.join()
            if model_container['error'] is not None:
              raise model_container['error']
            model = model_container['model']
            t0, t1 = model_container['load_time']
            _emit_timing("model_load", None, t0, t1)
          if keep_byproducts and write_queue is not None:
            out_dir = demix_dir_actual / 'htdemucs' / path.stem
            spec_path = spec_dir_actual / f'{path.stem}.npy'
            spec_np = np.array(spec, copy=False)
            stems_np = {stem: np.array(audio, copy=False) for stem, audio in stems.items()}
            write_queue.put((stems_np, out_dir, separator.samplerate, spec_np, spec_path))

          timings = {}
          result = run_inference_mlx_spec(
            path=path,
            spec=spec,
            model=model,
            include_activations=include_activations,
            include_embeddings=include_embeddings,
            compile_forward=mlx_compile,
            timings=timings,
          )
          if "nn" in timings:
            _emit_timing("nn", path, *timings["nn"])
          if "postprocess" in timings:
            _emit_timing("postprocess", path, *timings["postprocess"])
          for key in (
            "metrical_prep",
            "metrical_dbn",
            "functional_probs",
            "functional_local_maxima",
            "functional_boundaries",
          ):
            if key in timings:
              _emit_timing(key, path, *timings[key])
          if timings_embed:
            result.activations = result.activations or {}
            result.activations["timings"] = {
              "nn": (timings["nn"][1] - timings["nn"][0]) if "nn" in timings else None,
              "postprocess": (timings["postprocess"][1] - timings["postprocess"][0]) if "postprocess" in timings else None,
            }

          _emit_summary(path, timings)

          if out_dir is not None:
            t0 = time.perf_counter()
            save_results(result, out_dir)
            t1 = time.perf_counter()
            _emit_timing("save", path, t0, t1)

          results.append(result)
      else:
        spec_paths_for_infer = [spec_map[path.stem] for path in todo_paths]
        new_results = _run_mlx_inference(
          todo_paths=todo_paths,
          spec_paths_for_infer=spec_paths_for_infer,
          model_container=model_container,
          model_loader_thread=model_loader_thread,
          model_name=model_name,
          mlx_weights_dir=mlx_weights_dir,
          mlx_weights_path=mlx_weights_path,
          mlx_config_path=mlx_config_path,
          include_activations=include_activations,
          include_embeddings=include_embeddings,
          timings_embed=timings_embed,
          mlx_batch_size=mlx_batch_size,
          mlx_compile=mlx_compile,
          ensemble_parallel=ensemble_parallel,
          out_dir=out_dir,
          _emit_timing=_emit_timing,
          _emit_summary=_emit_summary if timings_summary else None,
        )
        results.extend(new_results)

    results = sorted(results, key=lambda result: paths.index(result.path))

    if visualize:
      from .visualize import visualize as _visualize
      t0 = time.perf_counter()
      if visualize is True:
        visualize = './viz'
      viz_results = results
      t_prep0 = time.perf_counter()
      if not overwrite_viz and visualize is not None:
        viz_dir = mkpath(visualize)
        viz_results = [
          result for result in results
          if not (viz_dir / f"{result.path.stem}.pdf").is_file()
        ]
      t_prep1 = time.perf_counter()
      _emit_timing("visualize_prep", None, t_prep0, t_prep1)
      _visualize(viz_results, out_dir=visualize, multiprocess=multiprocess)
      t1 = time.perf_counter()
      _emit_timing("visualize", None, t0, t1)
      print(f'=> Plots are successfully saved to {visualize}')

    if sonify:
      from .sonify import sonify as _sonify
      t0 = time.perf_counter()
      if sonify is True:
        sonify = './sonif'
      sonif_results = results
      if not overwrite_sonify and sonify is not None:
        sonif_dir = mkpath(sonify)
        sonif_results = [
          result for result in results
          if not (sonif_dir / f"{result.path.stem}.sonif{result.path.suffix}").is_file()
        ]
      _sonify(sonif_results, out_dir=sonify, multiprocess=multiprocess)
      t1 = time.perf_counter()
      _emit_timing("sonify", None, t0, t1)
      print(f'=> Sonified tracks are successfully saved to {sonify}')

    if not keep_byproducts and temp_dir is not None:
      shutil.rmtree(temp_dir, ignore_errors=True)
    elif not keep_byproducts:
      if overwrite_demix or overwrite_spec:
        for path in demix_paths:
          for stem in ['bass', 'drums', 'other', 'vocals']:
            (path / f'{stem}.wav').unlink(missing_ok=True)
          rmdir_if_empty(path)
        rmdir_if_empty(demix_dir_actual / 'htdemucs')
        rmdir_if_empty(demix_dir_actual)

        for path in spec_paths:
          path.unlink(missing_ok=True)
        rmdir_if_empty(spec_dir_actual)

    if timings_viz_path is not None:
      timings_viz_pending = True
  finally:
    if writer_thread is not None and write_queue is not None:
      write_queue.put(None)
      writer_thread.join()
    if timings_handle is not None:
      timings_handle.close()
    if summary_handle is not None:
      summary_handle.close()

  if timings_summary and cache_stats_before is not None:
    cache_stats_after = None  # Model cache stats not implemented
    print(
      "=> cache_summary "
      f"model_hit={cache_stats_after['model_hits'] - cache_stats_before['model_hits']} "
      f"model_miss={cache_stats_after['model_misses'] - cache_stats_before['model_misses']} "
      f"ensemble_hit={cache_stats_after['ensemble_hits'] - cache_stats_before['ensemble_hits']} "
      f"ensemble_miss={cache_stats_after['ensemble_misses'] - cache_stats_before['ensemble_misses']} "
      f"model_cache={cache_stats_after['model_cache_size']} "
      f"ensemble_cache={cache_stats_after['ensemble_cache_size']}"
    )
    try:
      from .postprocessing.dbn_native import selected_viterbi_backend
      print(f"=> dbn_backend {selected_viterbi_backend()}")
    except Exception:
      pass

  if timings_json_summary_path is not None and timing_records is not None:
    stage_totals: Dict[str, float] = {}
    per_track_totals: Dict[str, Dict[str, float]] = {}
    for rec in timing_records:
      stage = str(rec.get("stage"))
      duration = float(rec.get("duration", 0.0))
      stage_totals[stage] = stage_totals.get(stage, 0.0) + duration
      track = rec.get("track")
      if track is not None:
        per_track = per_track_totals.setdefault(str(track), {})
        per_track[stage] = per_track.get(stage, 0.0) + duration

    analyze_end = time.perf_counter()
    summary_payload: Dict[str, Any] = {
      "model": model_name,
      "device": device,
      "dbn_backend": os.environ.get("ALLIN1_DBN_BACKEND", "auto"),
      "num_tracks": len(paths),
      "wall_time_s": round(float(analyze_end - analyze_start), 2),
      "stage_totals_s": {k: round(v, 2) for k, v in stage_totals.items()},
      "per_track_stage_totals_s": {
        track: {stage: round(duration, 2) for stage, duration in stages.items()}
        for track, stages in per_track_totals.items()
      },
    }
    timings_json_summary_path.parent.mkdir(parents=True, exist_ok=True)
    timings_json_summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True))

  if timings_viz_pending:
    if timings_path is None:
      raise ValueError("timings_viz_path requires timings_path to be set.")
    from .timings_viz import visualize_timings
    visualize_timings(Path(timings_path), Path(timings_viz_path))
  if not return_list:
    return results[0]
  return results
