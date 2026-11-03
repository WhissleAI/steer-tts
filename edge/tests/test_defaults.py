"""Per-voice default profile (config.json "default_controls"): pace multiplier, offsets on set controls, opt-out."""
import json
from pathlib import Path

import numpy as np
import pytest

from steertts_edge.config import profiled

TEXT = "The meeting has been moved to Thursday afternoon at three."
CFG = {"default_controls": {"version": 1, "voices": {"v1": {"pace": 1.2, "ctrl": {"rate": 0.5}}}, "*": {"pace": 1.05},
                            "clone": {"pace": 0.9}}}


def test_rules():
    assert profiled(CFG, "v1", None, {"rate": 1.0, "loud": -1.0}, 1.0) == ({"rate": 1.5, "loud": -1.0}, 1.2)
    assert profiled(CFG, "v1", None, None, 1.0) == (None, 1.2)  # unset controls stay unset
    assert profiled(CFG, "v1", None, {"rate": 4.8}, 4.0) == ({"rate": 5.0}, 4.0)  # clipped to the request ranges
    assert profiled(CFG, "v1", None, {"rate": 9}, 1.0)[0] == {"rate": 9}  # out of range: left to the request check
    assert profiled(CFG, "other", None, None, 1.0) == (None, 1.05)
    assert profiled(CFG, "v1", np.zeros(512), None, 1.0) == (None, 0.9)
    assert profiled({}, "v1", None, {"rate": 1}, 1.3) == ({"rate": 1}, 1.3)  # no table: unchanged


def test_package_table(model_dir):
    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    dc = cfg.get("default_controls")
    if dc is None:
        pytest.skip("config.json has no default_controls")
    assert dc["voices"][cfg["default_voice"]]["pace"] == pytest.approx(1.10)
    assert all(not p.get("ctrl") for p in dc["voices"].values())  # pace only: tags keep choosing the rate
    assert set(dc["voices"]) <= set(tts_voices(model_dir))


def tts_voices(model_dir):
    return [v["name"] for v in json.loads((Path(model_dir) / "voices.json").read_text())["voices"]]


def test_profile_is_a_pace_multiplier(tts):
    pace = profiled(tts.cfg, tts.default_voice, None, None, 1.0)[1]
    if pace == 1.0:
        pytest.skip("no profile for the default voice")
    default = tts(TEXT)
    assert np.array_equal(default, tts(TEXT, pace=pace, default_controls=False))
    assert len(default) < len(tts(TEXT, default_controls=False))
    s = np.concatenate([c.audio for c in tts.stream(TEXT)])
    assert np.array_equal(s, np.concatenate([c.audio for c in tts.stream(TEXT, pace=pace, default_controls=False)]))


def test_other_voices_unchanged(tts):
    v = next(v for v in tts.voices if v != tts.default_voice and profiled(tts.cfg, v, None, None, 1.0)[1] == 1.0)
    assert np.array_equal(tts(TEXT, voice=v), tts(TEXT, voice=v, default_controls=False))
