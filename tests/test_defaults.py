"""Per-voice default profile (config.json "default_controls"): pace multiplier, offsets on set controls, opt-out."""
import json
from pathlib import Path

import numpy as np
import pytest

from steertts.defaults import apply_profile, check_table, voice_profile

TEXT = "The meeting has been moved to Thursday afternoon at three."
CFG = {"default_controls": {"version": 1, "voices": {"v1": {"pace": 1.2, "ctrl": {"rate": 0.5}}}, "*": {"pace": 1.05},
                            "clone": {"pace": 0.9}}}


# ------------------------------------------------------------------ no model needed
def test_table_lookup():
    assert voice_profile(CFG, "v1") == {"pace": 1.2, "ctrl": {"rate": 0.5}}
    assert voice_profile(CFG, "other") == {"pace": 1.05, "ctrl": {}}
    assert voice_profile(CFG, "v1", clone=True) == {"pace": 0.9, "ctrl": {}}
    assert voice_profile({}, "v1") == {"pace": 1.0, "ctrl": {}}  # no table: identity


def test_apply_profile_rules():
    p = voice_profile(CFG, "v1")
    assert apply_profile(p, {"rate": 1.0, "loud": -1.0}, 1.0) == ({"rate": 1.5, "loud": -1.0}, 1.2)
    assert apply_profile(p, {}, 1.0) == ({}, 1.2)  # unset controls stay unset
    assert apply_profile(p, {"rate": 4.8}, 4.0) == ({"rate": 5.0}, 4.0)  # clipped to the request ranges


def test_check_table():
    dims = ["rate", "loud"]
    check_table(CFG, {"v1"}, dims)
    for bad in ({"voices": {"nope": {"pace": 1.1}}}, {"voices": {"v1": {"pace": 3.0}}},
                {"voices": {"v1": {"ctrl": {"speed": 1}}}}, {"voices": {"v1": {"ctrl": {"rate": 2.5}}}}):
        with pytest.raises(ValueError):
            check_table({"default_controls": bad}, {"v1"}, dims)


def test_package_table(model_dir):
    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    dc = cfg.get("default_controls")
    if dc is None:
        pytest.skip("config.json has no default_controls")
    assert dc["voices"][cfg["default_voice"]]["pace"] == pytest.approx(1.10)
    assert all(not p.get("ctrl") for p in dc["voices"].values())  # pace only: tags keep choosing the rate


# ------------------------------------------------------------------ model
def test_profile_is_a_pace_multiplier(tts):
    if tts.default_profile()["pace"] == 1.0:
        pytest.skip("no profile for the default voice")
    pace = tts.default_profile()["pace"]
    default = tts(TEXT)
    hand = tts.synthesize({"text": TEXT, "pace": pace, "default_controls": False})
    raw = tts.synthesize({"text": TEXT, "default_controls": False})
    assert np.array_equal(default, hand)
    assert len(default) < len(raw)


def test_resolve_folds_the_profile(tts):
    req = tts.resolve({"text": TEXT, "ctrl": {"rate": 1}})
    assert req["default_controls"] is False and req["pace"] == pytest.approx(tts.default_profile()["pace"])
    assert np.array_equal(tts.synthesize(req), tts.synthesize({"text": TEXT, "ctrl": {"rate": 1}}))


def test_other_voices_unchanged(tts):
    v = next(v for v in tts.voices if v != tts.default_voice and tts.default_profile(v)["pace"] == 1.0)
    assert np.array_equal(tts(TEXT, voice=v), tts.synthesize({"text": TEXT, "voice": v, "default_controls": False}))
