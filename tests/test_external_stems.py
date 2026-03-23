import importlib
from pathlib import Path

import numpy as np

from allin1_mlx.typings import AnalysisResult, Segment

analyze_module = importlib.import_module("allin1_mlx.analyze")


def test_prepare_external_stems_collapses_bs_roformer_six_stems(monkeypatch, tmp_path):
  audio_path = tmp_path / "song.wav"
  stem_names = ("bass", "drums", "other", "vocals", "guitar", "piano")
  for stem_name in stem_names:
    (tmp_path / f"song_({stem_name}).wav").write_bytes(b"")

  stem_audio = {
    "bass": np.full(8, 1.0, dtype=np.float32),
    "drums": np.full(8, 2.0, dtype=np.float32),
    "other": np.full(8, 3.0, dtype=np.float32),
    "vocals": np.full(8, 4.0, dtype=np.float32),
    "guitar": np.full(8, 5.0, dtype=np.float32),
    "piano": np.full(8, 6.0, dtype=np.float32),
  }
  monkeypatch.setattr(
    analyze_module,
    "_load_external_stem_audio",
    lambda paths: (stem_audio, 44100),
  )

  stems, sample_rate, mode = analyze_module._prepare_external_stems(tmp_path, audio_path)

  assert sample_rate == 44100
  assert mode == "external_6_to_4"
  assert sorted(stems) == ["bass", "drums", "other", "vocals"]
  np.testing.assert_array_equal(stems["other"], np.full(8, 14.0, dtype=np.float32))


def test_analyze_uses_external_stems_and_skips_demix(monkeypatch, tmp_path):
  audio_path = tmp_path / "song.wav"
  audio_path.write_bytes(b"RIFF")

  captured = {}
  result = AnalysisResult(
    path=audio_path,
    bpm=120,
    beats=[0.0, 0.5],
    downbeats=[0.0],
    beat_positions=[1, 2],
    segments=[Segment(start=0.0, end=1.0, label="intro")],
  )

  monkeypatch.setattr(analyze_module, "demix", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("demix should not run")))
  monkeypatch.setattr(analyze_module, "load_pretrained_model_mlx", lambda **kwargs: object())
  monkeypatch.setattr(
    analyze_module,
    "_prepare_external_stems",
    lambda stems_dir, path: (
      {
        "bass": np.zeros(16, dtype=np.float32),
        "drums": np.zeros(16, dtype=np.float32),
        "other": np.zeros(16, dtype=np.float32),
        "vocals": np.zeros(16, dtype=np.float32),
      },
      44100,
      "external_4_stem",
    ),
  )

  def fake_spectrogram_from_stems(stems, sample_rate, **kwargs):
    captured["stems"] = stems
    captured["sample_rate"] = sample_rate
    return np.zeros((4, 8), dtype=np.float32)

  monkeypatch.setattr(analyze_module, "spectrogram_from_stems", fake_spectrogram_from_stems)
  monkeypatch.setattr(analyze_module, "run_inference_mlx_spec", lambda **kwargs: result)

  resolved = analyze_module.analyze(audio_path, stems_dir=tmp_path, mlx_in_memory=False)

  assert resolved == result
  assert captured["sample_rate"] == 44100
  assert sorted(captured["stems"]) == ["bass", "drums", "other", "vocals"]
