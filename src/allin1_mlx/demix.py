import shutil
from pathlib import Path
from typing import List


def demix(
  paths: List[Path],
  demix_dir: Path,
  overwrite: bool = False,
):
  """Demixes audio files into source-separated stems using demucs-mlx."""

  todos = []
  demix_paths = []
  for path in paths:
    out_dir = demix_dir / 'htdemucs' / path.stem
    demix_paths.append(out_dir)
    if out_dir.is_dir():
      if overwrite:
        shutil.rmtree(out_dir, ignore_errors=True)
        todos.append(path)
        continue
      if (
        (out_dir / 'bass.wav').is_file() and
        (out_dir / 'drums.wav').is_file() and
        (out_dir / 'other.wav').is_file() and
        (out_dir / 'vocals.wav').is_file()
      ):
        continue
    todos.append(path)

  if overwrite:
    existing = 0
  else:
    existing = len(paths) - len(todos)
  print(f'=> Found {existing} tracks already demixed, {len(todos)} to demix.')

  if todos:
    try:
      from demucs_mlx.api import Separator, save_audio
    except Exception as exc:
      raise ImportError(
        "demucs-mlx is not available. Install it with `uv pip install demucs-mlx`."
      ) from exc

    separator = Separator(model="htdemucs", progress=False)
    for path in todos:
      _, stems = separator.separate_audio_file(path)
      output_subdir = demix_dir / 'htdemucs' / path.stem
      output_subdir.mkdir(parents=True, exist_ok=True)
      for stem, audio in stems.items():
        save_audio(audio, output_subdir / f"{stem}.wav", separator.samplerate)

  return demix_paths
