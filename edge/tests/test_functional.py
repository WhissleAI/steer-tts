"""Every voice, every tag, every control, determinism, errors."""
import numpy as np
import pytest

from conftest import SR, check_audio

SHORT = "The garden looks lovely this morning."
TEXT = "Please remember to bring the documents to the meeting on Thursday."


def test_all_voices(tts):
    durs = {}
    for v in tts.voices:
        durs[v] = check_audio(tts(SHORT, voice=v), 0.8, 6.0)
    d = np.array(list(durs.values()))
    assert d.max() / d.min() < 3.0, (min(durs, key=durs.get), max(durs, key=durs.get))


def test_all_tags(tts):
    base = len(tts(TEXT))
    for t in tts.tags:
        y = tts(TEXT, tag=t)
        check_audio(y, 1.0, 12.0, min_peak=0.005 if t == "whisper" else 0.02)
        assert 0.4 < len(y) / base < 2.5, (t, len(y) / base)


@pytest.mark.parametrize("dim", ["rate", "loud", "f0_med", "f0_iqr", "f0_end", "pause", "tilt", "voiced", "flat",
                                 "centroid", "cpps", "h1h2"])
def test_control_levels(tts, dim):
    ys = {z: tts(TEXT, ctrl={dim: z}) for z in (-2, 0, 2)}
    for z, y in ys.items():
        check_audio(y, 1.0, 12.0, min_peak=0.005)
    assert not np.array_equal(ys[-2], ys[2]), f"{dim}: -2 and +2 render identically"


def test_ctrl_zero_is_a_request(tts):
    # z=0 is an explicit "this voice's neutral" request (mask on) and may differ slightly from no control at all.
    a, b = tts(TEXT), tts(TEXT, ctrl={"rate": 0})
    assert np.isfinite(b).all() and abs(len(a) - len(b)) < 0.15 * len(a)


def test_control_direction_cheap(tts):
    """Proxies that need no measuring toolkit: rate -> duration, loud -> RMS."""
    n = {z: len(tts(TEXT, ctrl={"rate": z})) for z in (-2, 2)}
    assert n[2] < 0.85 * n[-2], n
    rms = {z: float(np.sqrt(np.mean(tts(TEXT, ctrl={"loud": z}) ** 2))) for z in (-2, 2)}
    assert rms[2] > 1.5 * rms[-2], rms
    assert len(tts(TEXT, pace=1.5)) < 0.8 * len(tts(TEXT))


def test_deterministic(tts):
    for kw in ({}, {"voice": tts.voices[17], "tag": "sadness", "ctrl": {"loud": -1}, "guidance": 1.5}):
        assert np.array_equal(tts(TEXT, **kw), tts(TEXT, **kw))


def test_guidance_and_word_shift(tts):
    check_audio(tts(TEXT, tag="anger", guidance=2.0), 1.0, 12.0)
    a, b = tts(TEXT), tts(TEXT, word_shift={2: {"pitch": 3, "energy": 4, "dur": 1.5}})
    assert len(b) > len(a) and not np.array_equal(a, b)


def test_errors(tts):
    with pytest.raises(ValueError, match="unknown voice 'voice_9999'"):
        tts(TEXT, voice="voice_9999")
    with pytest.raises(ValueError, match="unknown tag 'angry'.*known tags"):
        tts(TEXT, tag="angry")
    with pytest.raises(ValueError, match="unknown control 'speed'"):
        tts(TEXT, ctrl={"speed": 1})
    with pytest.raises(ValueError, match="outside"):
        tts(TEXT, ctrl={"rate": 9})
    with pytest.raises(ValueError, match="not available in the edge build"):
        tts(TEXT, prompt="angry and fast")
    with pytest.raises(ValueError, match="not available in the edge build"):
        tts(TEXT, ref_wav="x.wav")
    with pytest.raises(ValueError, match="unknown voice"):
        tts(TEXT, voice="voice_9999", markup=True)
    with pytest.raises(ValueError, match="unknown tag"):
        list(tts.stream(TEXT, tag="angry"))
    with pytest.raises(ValueError, match="word_shift"):
        tts(TEXT, markup=True, word_shift={0: {"pitch": 2}})
    with pytest.raises(ValueError, match="word_shift"):
        list(tts.stream(TEXT, word_shift={0: {"pitch": 2}}))


def test_tag_none_and_flow_compat(tts):
    a = tts(TEXT)
    assert np.array_equal(a, tts(TEXT, tag=None))
    assert np.array_equal(a, tts(TEXT, flow=True, seed=3))  # full-model args are accepted and ignored


def test_save_wav(tts, tmp_path):
    import wave
    y = tts("Saved to disk.")
    p = tmp_path / "x.wav"
    tts.save(y, p)
    with wave.open(str(p)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()) == (SR, 1, 2, len(y))
        pcm = np.frombuffer(w.readframes(len(y)), "<i2").astype(np.float32) / 32767
    assert np.abs(pcm - np.clip(y, -1, 1)).max() <= 1 / 32767 + 1e-7
