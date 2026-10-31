import numpy as np
import pytest

from steertts import SR

TEXT = "The meeting has been moved to Thursday afternoon at three."


def test_voices_tags(tts):
    assert len(tts.voices) > 1000 and all(v.startswith("voice_") for v in tts.voices)
    assert tts.default_voice in tts.voices and "whisper" in tts.tags and "none" in tts.tags


def test_basic_render_deterministic(tts):
    a = tts(TEXT)
    b = tts(TEXT, voice=tts.default_voice)
    assert a.dtype == np.float32 and 1.5 < len(a) / SR < 8 and np.isfinite(a).all() and np.abs(a).max() > 0.01
    assert np.array_equal(a, b)


def test_flow_render(tts_flow):
    a = tts_flow(TEXT, seed=0)
    assert np.isfinite(a).all() and np.array_equal(a, tts_flow(TEXT, seed=0))


def test_pace_and_rate(tts):
    n = len(tts(TEXT))
    assert len(tts(TEXT, pace=1.5)) < 0.8 * n
    assert len(tts(TEXT, ctrl={"rate": 2})) < len(tts(TEXT, ctrl={"rate": -2}))


@pytest.mark.parametrize("dim", ["loud", "f0_med", "h1h2"])
def test_control_readback_direction(tts, dim):
    pytest.importorskip("parselmouth")
    from steertts.measure import measure_z
    v = tts.default_voice
    z = {s: measure_z(tts(TEXT, ctrl={dim: s}), TEXT, tts.voice_stats(v), tts.pooled_sd)[dim] for s in (-2, 2)}
    assert z[2] - z[-2] > 1.0, z


def test_markup_stream(tts):
    chunks = list(tts.stream("First sentence here. [pause 300ms] *Second* one. <whisper>Third.</whisper>"))
    kinds = [("pause" if c.segment.pause_ms else c.segment.tag) for c in chunks]
    assert kinds == ["none", "pause", "none", "whisper"], kinds
    assert abs(len(chunks[1].audio) - int(0.3 * SR)) <= 1
    assert chunks[2].segment.word_shift  # emphasis became a per-word shift


def test_clone_from_reference(tts, tmp_path):
    pytest.importorskip("transformers")
    ref = tmp_path / "ref.wav"
    tts.save(tts("This is the reference recording for a cloning test, about four seconds long.",
                 voice=tts.voices[7]), ref)
    y = tts("Now the cloned voice says something else.", ref_wav=str(ref))
    assert np.isfinite(y).all() and len(y) > SR


def test_prompt_maps_to_request(tts):
    pytest.importorskip("transformers")
    r = tts.resolve({"text": "Hello.", "prompt": "whispering, slow and secretive"})
    assert "prompt" not in r and r["prompt_mapping"]["top_tags"] and r["tag"] in tts.tags
    assert set(r["ctrl"]) <= set(tts.controls)
