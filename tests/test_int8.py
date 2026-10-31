import numpy as np
import pytest

from steertts import SR

TEXT = "The meeting has been moved to Thursday afternoon at three."


@pytest.fixture(scope="session")
def tts8(model_dir):
    from pathlib import Path
    if not (Path(model_dir) / "int8" / "model.safetensors").exists():
        pytest.skip("no int8/ weights in the model directory")
    from steertts import SteerTTS
    from conftest import DEV
    return SteerTTS(model_dir, device=DEV, flow=False, precision="int8")


def test_int8_render(tts8):
    a = tts8(TEXT)
    assert a.dtype == np.float32 and 1.5 < len(a) / SR < 8 and np.isfinite(a).all() and np.abs(a).max() > 0.01
    assert np.array_equal(a, tts8(TEXT))


def test_int8_close_to_fp32(tts, tts8):
    assert abs(len(tts8(TEXT)) / len(tts(TEXT)) - 1) < 0.05
    sd8, sd = tts8.model.state_dict(), tts.model.state_dict()
    for k, w in sd.items():
        if w.is_floating_point() and w.numel():
            assert (sd8[k] - w).abs().max() <= w.abs().max() * 4e-3 + 1e-6, k


def test_int8_readback_direction(tts8):
    pytest.importorskip("parselmouth")
    from steertts.measure import measure_z
    v = tts8.default_voice
    z = {s: measure_z(tts8(TEXT, ctrl={"f0_med": s}), TEXT, tts8.voice_stats(v), tts8.pooled_sd)["f0_med"] for s in (-2, 2)}
    assert z[2] - z[-2] > 1.0, z


def test_round_int8_matches_packed(model_dir):
    """The in-place rounding used for the external models equals the packed int8 files (same rule)."""
    import torch
    from steertts.quant import load_packed, round_int8_
    from safetensors.torch import load_file
    sd = load_file(f"{model_dir}/model.safetensors")
    packed = load_packed(f"{model_dir}/int8/model.safetensors")
    m = torch.nn.Module()
    for k, w in sd.items():
        m.register_parameter(k.replace(".", "_"), torch.nn.Parameter(w.clone(), requires_grad=False))
    round_int8_(m)
    for k, w in sd.items():
        assert torch.equal(getattr(m, k.replace(".", "_")), packed[k]), k
