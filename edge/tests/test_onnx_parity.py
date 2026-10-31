"""The ONNX Runtime build renders what the PyTorch runtime renders: same durations (sample counts) and the same audio up
to float rounding (different kernels and summation order; the mel spectrograms agree to ~3e-5)."""
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

from conftest import make_tts  # noqa: E402

CASES = [
    dict(text="Hi, thanks for calling. I can help you reschedule your appointment today."),
    dict(text="Are you coming with us tonight?", voice="voice_1361", ctrl={"f0_end": 2, "f0_iqr": 1.5}),
    dict(text="The weather was cold, but the view from the top was worth every step.", ctrl={"rate": -1.5}),
    dict(text="Please keep this between the two of us.", tag="whisper"),
    dict(text="I really thought it would work this time.", tag="sadness", guidance=1.5),
    dict(text="Well, *that* was fast. [pause 400ms] <slow>Let me think.</slow> <whisper>Shh.</whisper>", markup=True),
    dict(text="It costs $3,499.99 on March 3rd, Dr. Zyxquorpian.", voice="voice_0047"),
    dict(text="Word shift on one word here.", word_shift={2: {"pitch": 3, "energy": 4, "dur": 1.3}}),
    dict(text=" ".join(["A long paragraph keeps going so the decoder sees well over a thousand frames."] * 4)),
]


@pytest.fixture(scope="module")
def pair(model_dir):
    return make_tts("torch", model_dir), make_tts("onnx", model_dir)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["text"][:24])
def test_same_audio(pair, case):
    a, b = (t(**case) for t in pair)
    assert len(a) == len(b)
    snr = 10 * np.log10((a.astype(np.float64) ** 2).sum() / max(((a - b).astype(np.float64) ** 2).sum(), 1e-20))
    assert snr > 40, f"SNR {snr:.1f} dB"


def test_same_mel(pair):
    a, b = (t.mel("The meeting moved to Thursday.") for t in pair)
    assert a.shape == b.shape and float(np.abs(a.numpy() - b).max()) < 1e-3
