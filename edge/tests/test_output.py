"""Output level (steertts_edge.output): calibration block, limiter, true-peak cap, the opt-out, streaming == whole."""
import json
from pathlib import Path

import numpy as np
import pytest

from steertts_edge.output import Output, check_level, lufs, soft_limit, true_peak_db

SR = 24000
TEXT = "The meeting has been moved to Thursday afternoon at three."


def test_soft_limit_curve():
    x = np.linspace(-2, 2, 4001, dtype=np.float32)
    y = soft_limit(x, -3.0, -1.0)
    knee = 10 ** (-3 / 20)
    assert np.abs(y).max() <= 10 ** (-1 / 20) + 1e-6
    assert np.array_equal(y[np.abs(x) <= knee], x[np.abs(x) <= knee]) and np.all(np.diff(y) >= 0)


def test_lufs_reference_tone():
    t = np.arange(10 * SR) / SR
    assert abs(lufs(0.1 * np.sin(2 * np.pi * 1000 * t), SR) + 23.0) < 0.1  # -20 dBFS 1 kHz sine = -23 LUFS


def test_levels_validated():
    check_level("clip")
    with pytest.raises(ValueError):
        check_level("max")
    with pytest.raises(ValueError):
        check_level("auto", -20)


def test_config_has_output_block(model_dir):
    cfg = json.load(open(Path(model_dir) / "config.json"))
    voices = [v["name"] for v in json.load(open(Path(model_dir) / "voices.json"))["voices"]]
    o = cfg["output"]
    assert o["target_lufs"] == -17.0 and o["true_peak_db"] == -1.0 and set(o["voice_gain_db"]) == set(voices)


def test_auto_vs_off(tts):
    raw, out = tts(TEXT, level="off"), tts(TEXT)
    g = tts.output.gain(tts.default_voice)
    assert len(raw) == len(out) and 4.0 < g < 14.0
    assert abs(lufs(out, SR) - lufs(raw, SR) - g) < 1.0
    assert true_peak_db(out) <= -1.0 + 1e-3 and abs(lufs(out, SR) + 17.0) < 2.5


def test_stream_equals_whole_clip(tts):
    text = "First sentence here. [pause 300ms] Second one, a little longer."
    whole = tts(text, markup=True)
    streamed = np.concatenate([c.audio for c in tts.stream(text)])
    k = float(np.dot(whole, streamed) / np.dot(streamed, streamed))
    assert len(whole) == len(streamed) and 0.9 < k <= 1.0 + 1e-6 and np.abs(whole - k * streamed).max() < 1e-4
    raw = np.concatenate([c.audio for c in tts.stream(text, level="off")])
    assert np.array_equal(raw, tts(text, markup=True, level="off"))


def test_loud_steering_survives(tts):
    lo, mid, hi = (lufs(tts(TEXT, ctrl={"loud": z}), SR) for z in (-2, 0, 2))
    assert lo + 2.0 < mid < hi - 1.0, (lo, mid, hi)


def test_constructor_opt_out(model_dir, tts):
    from conftest import make_tts
    raw = make_tts("onnx" if type(tts).__name__ == "OnnxTTS" else "torch", model_dir, level="off")
    assert np.array_equal(raw(TEXT), tts(TEXT, level="off"))
