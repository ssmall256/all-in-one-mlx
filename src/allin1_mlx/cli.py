import argparse
from pathlib import Path

from .analyze import analyze


def make_parser():
  cwd = Path.cwd()
  parser = argparse.ArgumentParser(prog='allin1-mlx')
  parser.add_argument('paths', nargs='+', type=Path, default=[], help='Path to tracks')
  parser.add_argument('-o', '--out-dir', type=Path, default=cwd / './struct',
                      help='Path to a directory to store analysis results (default: ./struct)')
  parser.add_argument('-v', '--visualize', action='store_true', default=False,
                      help='Save visualizations (default: False)')
  parser.add_argument('--viz-dir', type=str, default=cwd / 'viz',
                      help='Directory to save visualizations if -v is provided (default: ./viz)')
  parser.add_argument('-s', '--sonify', action='store_true', default=False,
                      help='Save sonifications (default: False)')
  parser.add_argument('--sonif-dir', type=str, default=cwd / 'sonif',
                      help='Directory to save sonifications if -s is provided (default: ./sonif)')
  parser.add_argument('-a', '--activ', action='store_true',
                      help='Save frame-level raw activations from sigmoid and softmax (default: False)')
  parser.add_argument('-e', '--embed', action='store_true',
                      help='Save frame-level embeddings (default: False)')
  parser.add_argument('-m', '--model', type=str, default='harmonix-all',
                      help='Name of the pretrained model to use (default: harmonix-all)')
  parser.add_argument('-d', '--device', type=str, default='mlx', choices=['mlx'],
                      help='Device to use (MLX-only build)')
  parser.add_argument('--mlx-weights-dir', type=Path, default=None,
                      help='Directory containing MLX weights/config (default lookup: ./mlx-weights)')
  parser.add_argument('--mlx-weights-path', type=Path, default=None,
                      help='Path to MLX weights file (overrides --mlx-weights-dir)')
  parser.add_argument('--mlx-config-path', type=Path, default=None,
                      help='Path to MLX config file (defaults next to weights file)')
  parser.add_argument('--mlx-batch-size', type=int, default=1,
                      help='Batch size for MLX inference (default: 1)')
  parser.add_argument('--mlx-compile', action='store_true',
                      help='Enable mx.compile for MLX model forward (default behavior)')
  parser.add_argument('--no-mlx-compile', action='store_true',
                      help='Disable mx.compile for MLX model forward')
  parser.add_argument('--mlx-in-memory', action='store_true',
                      help='Demix and compute spectrograms in-memory for MLX (default behavior)')
  parser.add_argument('--no-mlx-in-memory', action='store_true',
                      help='Disable in-memory MLX demix/spec pipeline')
  parser.add_argument('--ensemble-parallel', action='store_true',
                      help='Enable parallel ensemble inference (default: True)')
  parser.add_argument('--no-ensemble-parallel', action='store_true',
                      help='Disable parallel ensemble inference')
  parser.add_argument('--timings-path', type=Path, default=None,
                      help='Write JSONL timings to this path (default: None)')
  parser.add_argument('--timings-viz-path', type=Path, default=None,
                      help='Write timing visualization to this path (requires --timings-path)')
  parser.add_argument('--timings-embed', action='store_true',
                      help='Embed timing info in activations output (default: False)')
  parser.add_argument('-k', '--keep-byproducts', action='store_true',
                      help='Keep demixed audio files and spectrograms (default: False)')
  parser.add_argument('--demix-dir', type=Path, default=cwd / 'demix',
                      help='Path to a directory to store demixed tracks (default: ./demix)')
  parser.add_argument('--spec-dir', type=Path, default=cwd / 'spec',
                      help='Path to a directory to store spectrograms (default: ./spec)')
  parser.add_argument('--spec-backend', type=str, choices=['mlx', 'mlx_fast'], default=None,
                      help='Spectrogram backend to use (default: mlx_fast)')
  parser.add_argument('--spec-check', action='store_true',
                      help='Compare mlx_fast spectrograms to mlx once and report max/mean diff (default: False)')
  parser.add_argument('--overwrite', nargs='?', const='all', default=None,
                      help='Overwrite stages: all or csv (demix,spec,json,viz,sonify)')
  parser.add_argument('--no-multiprocess', action='store_true', default=False,
                      help='Disable multiprocessing (default: False)')

  return parser


def main():
  parser = make_parser()
  args = parser.parse_args()

  if not args.paths:
    raise ValueError('At least one path must be specified.')

  assert args.out_dir is not None, 'Output directory must be specified with --out-dir'

  if args.device != "mlx":
    raise ValueError("This MLX-only build supports only --device mlx.")

  if args.spec_backend is None:
    args.spec_backend = "mlx_fast"
  if args.mlx_compile and args.no_mlx_compile:
    raise ValueError("Use only one of --mlx-compile or --no-mlx-compile.")
  if args.mlx_compile:
    mlx_compile = True
  elif args.no_mlx_compile:
    mlx_compile = False
  else:
    mlx_compile = True
  if args.mlx_in_memory and args.no_mlx_in_memory:
    raise ValueError("Use only one of --mlx-in-memory or --no-mlx-in-memory.")
  if args.mlx_in_memory:
    mlx_in_memory = True
  elif args.no_mlx_in_memory:
    mlx_in_memory = False
  else:
    mlx_in_memory = True
  if args.ensemble_parallel and args.no_ensemble_parallel:
    raise ValueError("Use only one of --ensemble-parallel or --no-ensemble-parallel.")
  if args.ensemble_parallel:
    ensemble_parallel = True
  elif args.no_ensemble_parallel:
    ensemble_parallel = False
  else:
    ensemble_parallel = True

  analyze(
    paths=args.paths,
    out_dir=args.out_dir,
    visualize=args.viz_dir if args.visualize else False,
    sonify=args.sonif_dir if args.sonify else False,
    model=args.model,
    device=args.device,
    include_activations=args.activ,
    include_embeddings=args.embed,
    demix_dir=args.demix_dir,
    spec_dir=args.spec_dir,
    spec_backend=args.spec_backend,
    keep_byproducts=args.keep_byproducts,
    overwrite=args.overwrite,
    multiprocess=not args.no_multiprocess,
    mlx_weights_dir=args.mlx_weights_dir,
    mlx_weights_path=args.mlx_weights_path,
    mlx_config_path=args.mlx_config_path,
    mlx_batch_size=args.mlx_batch_size,
    mlx_compile=mlx_compile,
    mlx_in_memory=mlx_in_memory,
    ensemble_parallel=ensemble_parallel,
    spec_check=args.spec_check,
    timings_path=args.timings_path,
    timings_embed=args.timings_embed,
    timings_viz_path=args.timings_viz_path,
  )

  print(f'=> Analysis results are successfully saved to {args.out_dir}')


if __name__ == '__main__':
  main()
