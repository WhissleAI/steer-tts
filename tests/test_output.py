"""Output level (steertts.output): calibration block, limiter, true-peak cap, the opt-out, streaming == whole clip."""
import json
from pathlib import Path

import numpy as np
import pytest

from steertts import SR
from steertts.output import Output, check_level, lufs, soft_limit, true_peak_db

TEXT = "The meeting has been moved to Thursday afternoon at three."


# ------------------------------------------------------------------ no model needed
def test_soft_limit_curve():
    x = np.linspace(-2, 2, 4001, dtype=np.float32)
    y = soft_limit(x, -3.0, -1.0)
    knee, ceil = 10 ** (-3 / 20), 10 ** (-1 / 20)
    assert np.abs(y).max() <= ceil + 1e-6
    small = np.abs(x) <= knee
    assert np.array_equal(y[small], x[small])  # untouched below the knee
    assert np.all(np.diff(y) >= 0)  # monotonic, no fold-over


def test_lufs_reference_tone():
    t = np.arange(10 * SR) / SR
    x = (10 ** (-20 / 20)) * np.sin(2 * np.pi * 1000 * t)  # 1 kHz sine at -20 dBFS peak = -23.0 LUFS (BS.1770)
    assert abs(lufs(x, SR) - (-23.0)) < 0.1


def test_levels_validated():
    check_level("auto"); check_level("off", 3.0)
    with pytest.raises(ValueError):
        check_level("loud")
    with pytest.raises(ValueError):
        check_level("auto", 30)


def test_clip_mode_hits_target_and_cap():
    rng = np.random.default_rng(0)
    x = (0.05 * rng.standard_normal(3 * SR)).astype(np.float32)
    o = Output({"output": {"target_lufs": -17.0, "true_peak_db": -1.0}}, SR)
    y = o.clip(x, level="clip")
    assert abs(lufs(y, SR) + 17.0) < 0.7 and true_peak_db(y) <= -1.0 + 1e-3
    assert o.clip(x, level="off") is x


# ------------------------------------------------------------------ with the model
def test_config_has_output_block(model_dir):
    cfg = json.load(open(Path(model_dir) / "config.json"))
    o = cfg["output"]
    assert o["target_lufs"] == -17.0 and o["true_peak_db"] == -1.0
    assert set(o["voice_gain_db"]) == set(cfg["voices_order"])


def test_auto_vs_off(tts):
    raw, out = tts(TEXT, level="off"), tts(TEXT)
    assert len(raw) == len(out)
    g = tts.output.gain(tts.default_voice)
    assert 4.0 < g < 14.0
    assert abs(lufs(out, SR) - lufs(raw, SR) - g) < 1.0  # the limiter touches only the peaks
    assert true_peak_db(out) <= -1.0 + 1e-3
    assert abs(lufs(out, SR) - (-17.0)) < 2.5  # neutral speech near the target


def test_stream_equals_whole_clip(tts):
    text = "First sentence here. [pause 300ms] Second one, a little longer."
    whole = tts(text, markup=True)
    streamed = np.concatenate([c.audio for c in tts.stream(text)])
    assert len(whole) == len(streamed)
    # equal up to the whole clip's true-peak guard (a uniform scale <= 1)
    k = float(np.dot(whole, streamed) / np.dot(streamed, streamed))
    assert 0.9 < k <= 1.0 + 1e-6 and np.abs(whole - k * streamed).max() < 1e-4
    raw = np.concatenate([c.audio for c in tts.stream(text, level="off")])
    assert np.array_equal(raw, tts(text, markup=True, level="off"))


def test_loud_steering_survives(tts):
    lo, mid, hi = (lufs(tts(TEXT, ctrl={"loud": z}), SR) for z in (-2, 0, 2))
    assert lo + 2.0 < mid < hi - 1.0, (lo, mid, hi)


def test_gain_override(tts):
    a = tts(TEXT, gain_db=0.0)
    raw = tts(TEXT, level="off")
    assert abs(lufs(a, SR) - lufs(raw, SR)) < 0.3
