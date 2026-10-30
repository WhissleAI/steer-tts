"""Recorded sound events: [laugh] [cough] ... rendered from the package's event bank (events.safetensors)."""
from pathlib import Path

import numpy as np
import pytest

from conftest import SR, check_audio
from steertts_edge.events import EventBank, pitch_shift, wav_mel


def _bank(model_dir):
    b = EventBank.find(model_dir)
    if b is None:
        pytest.skip("no events.safetensors in the package")
    return b


def test_pitch_shift():
    mel = np.full((100, 4), -8.0, np.float32)
    mel[30] = 0.0
    assert np.array_equal(pitch_shift(mel, 1.0), mel)
    up = pitch_shift(mel, 2 ** (4 / 12))
    assert int(np.argmax(up[:, 0])) > 30  # a 4-semitone shift up moves the peak to a higher band


def test_bank(model_dir):
    b = _bank(model_dir)
    for t in ("laugh", "cough", "sigh", "sneeze", "throat"):
        assert t in b
        m = b.mel(t, "female", 220.0, ref_loud=-2.0, seed=1)
        assert m.shape[0] == 100 and m.shape[1] > 30 and np.isfinite(m).all()
        assert abs(EventBank.loudness(m[:, 7:-7]) - (-2.0 - 2 / 20 * np.log(10))) < 0.05


def test_event_in_text(tts, model_dir):
    _bank(model_dir)
    text = "Oh no, I did it again. [laugh] Okay, let me start over."
    chunks = list(tts.stream(text))
    ev = [c for c in chunks if c.segment.event]
    assert [c.segment.event for c in ev] == ["laugh"]
    y = ev[0].audio
    check_audio(y, min_s=0.3, max_s=3.0)
    # the event sits near the speech level (2 dB below by design; the vocoder adds a little spread)
    sp = np.concatenate([c.audio for c in chunks if not c.segment.event and len(c.audio)])
    d = EventBank.loudness(wav_mel(y)) - EventBank.loudness(wav_mel(sp))
    assert -1.2 < d < 0.6, d
    # a plain call (no markup=True) renders the same event chunk
    full = tts(text)
    assert abs(len(full) - sum(len(c.audio) for c in chunks)) < SR // 100


def test_event_first_and_only(tts, model_dir):
    _bank(model_dir)
    chunks = list(tts.stream("[cough] Sorry about that."))
    assert chunks[0].segment.event == "cough"  # waits for the first sentence's level, keeps its place
    check_audio(tts("[sigh]"), min_s=0.3, max_s=3.0)


def test_unknown_token_still_silent(tts, model_dir):
    b = _bank(model_dir)
    missing = next(t for t in ("thunder", "reserved0") if t not in b)
    chunks = list(tts.stream(f"Listen. [{missing}] Did you hear that?"))
    assert not any(c.segment.event for c in chunks)
