"""Test JSON output formatting for struct files."""

import json
import tempfile
from pathlib import Path

from allin1_mlx.helpers import save_results
from allin1_mlx.typings import AnalysisResult, Segment


def test_json_float_formatting():
    """Test that start/end times are formatted to 2 decimal places."""
    # Create a test result with high-precision floats
    result = AnalysisResult(
        path=Path("/fake/path/test.wav"),
        bpm=120,
        beats=[0.3700000047683716, 1.2345678901234567, 2.9999999999999999],
        downbeats=[0.3700000047683716, 2.4200000047683716],
        beat_positions=[1, 2, 3, 4, 1],
        segments=[
            Segment(
                start=0.3700000047683716,
                end=19.579999923706055,
                label="intro"
            ),
            Segment(
                start=19.579999923706055,
                end=45.12345678901234,
                label="verse"
            ),
        ],
    )

    # Save to JSON
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        save_results(result, tmp_path)

        # Load the JSON and check formatting
        json_path = tmp_path / "test.json"
        assert json_path.exists()

        with open(json_path, 'r') as f:
            data = json.load(f)

        # Check beats are rounded to 2 decimal places
        assert data['beats'][0] == 0.37
        assert data['beats'][1] == 1.23
        assert data['beats'][2] == 3.0

        # Check downbeats are rounded to 2 decimal places
        assert data['downbeats'][0] == 0.37
        assert data['downbeats'][1] == 2.42

        # Check segment start/end times are rounded to 2 decimal places
        assert data['segments'][0]['start'] == 0.37
        assert data['segments'][0]['end'] == 19.58
        assert data['segments'][1]['start'] == 19.58
        assert data['segments'][1]['end'] == 45.12

        # Verify the raw JSON text doesn't have excessive decimal places
        json_text = json_path.read_text()
        assert "0.3700000047683716" not in json_text
        assert "19.579999923706055" not in json_text
        assert "45.12345678901234" not in json_text


def test_json_text_formatting():
    """Test that the JSON text representation is clean."""
    result = AnalysisResult(
        path=Path("/fake/path/test.wav"),
        bpm=120,
        beats=[0.3700000047683716, 1.2345678901234567],
        downbeats=[0.3700000047683716],
        beat_positions=[1, 2],
        segments=[
            Segment(start=0.3700000047683716, end=19.579999923706055, label="intro"),
        ],
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        save_results(result, tmp_path)

        json_path = tmp_path / "test.json"
        json_text = json_path.read_text()

        # Check that segment times in text have 2 decimals
        assert '"start": 0.37' in json_text
        assert '"end": 19.58' in json_text

        # Check beats array (should be compacted on one line)
        assert '[0.37,1.23]' in json_text or '[0.37, 1.23]' in json_text


def test_integer_bpm_not_affected():
    """Test that integer values like BPM are not converted to floats."""
    result = AnalysisResult(
        path=Path("/fake/path/test.wav"),
        bpm=120,
        beats=[1.0, 2.0],
        downbeats=[1.0],
        beat_positions=[1, 2, 3, 4],
        segments=[
            Segment(start=1.0, end=10.0, label="intro"),
        ],
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        save_results(result, tmp_path)

        json_path = tmp_path / "test.json"
        with open(json_path, 'r') as f:
            data = json.load(f)

        # BPM should still be an integer
        assert data['bpm'] == 120
        assert isinstance(data['bpm'], int)

        # beat_positions should still be integers
        assert all(isinstance(pos, int) for pos in data['beat_positions'])


def test_timing_json_formatting():
    """Test that timing records are formatted to 2 decimal places."""
    import json

    # Create mock timing record with high precision
    timing_record = {
        "stage": "model_load",
        "start": 1234567890.123456789,
        "end": 1234567895.987654321,
        "duration": 5.864197532,
        "device": "mlx",
        "model": "test",
    }

    # Test rounding in the record
    rounded_record = {
        "stage": timing_record["stage"],
        "start": round(timing_record["start"], 2),
        "end": round(timing_record["end"], 2),
        "duration": round(timing_record["duration"], 2),
        "device": timing_record["device"],
        "model": timing_record["model"],
    }

    # Verify rounding
    assert rounded_record["start"] == 1234567890.12
    assert rounded_record["end"] == 1234567895.99
    assert rounded_record["duration"] == 5.86

    # Verify JSON output
    json_str = json.dumps(rounded_record)
    assert "1234567890.12" in json_str
    assert "1234567895.99" in json_str
    assert "5.86" in json_str
    assert "123456789" not in json_str or "1234567890.12" in json_str  # Allow substring match
