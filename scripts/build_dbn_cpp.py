#!/usr/bin/env python3
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


def main() -> None:
  root = Path(__file__).resolve().parents[1]
  src = root / "src" / "allin1_mlx" / "postprocessing" / "_viterbi_cpp.cpp"
  if not src.is_file():
    raise FileNotFoundError(f"Missing source file: {src}")

  ext_modules = [
    Pybind11Extension(
      "allin1_mlx.postprocessing._viterbi_cpp",
      [str(src)],
      cxx_std=17,
      extra_compile_args=["-O3"],
    )
  ]

  setup(
    name="allin1-mlx-dbndbn-cpp",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
  )


if __name__ == "__main__":
  main()
