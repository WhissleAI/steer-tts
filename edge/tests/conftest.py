import os
from pathlib import Path

import numpy as np
import pytest

# The package directory (weights). Default: this repo. Set STEERTTS_EDGE_MODEL to test an installed package against
# a directory elsewhere (run pytest from outside the source tree so `steertts_edge` comes from site-packages).
MODEL = os.environ.get("STEERTTS_EDGE_MODEL") or str(Path(__file__).resolve().parents[1])
SR = 24000


@pytest.fixture(scope="session")
def model_dir():
    if not (Path(MODEL) / "config.json").exists():
        pytest.skip(f"no package directory at {MODEL} (set STEERTTS_EDGE_MODEL)")
    return MODEL


def _runtimes():
    """Runtimes under test: STEERTTS_EDGE_RUNTIMES=onnx,torch (default: every one whose dependency is installed)."""
    want = os.environ.get("STEERTTS_EDGE_RUNTIMES")
    if want:
        return want.split(",")
    out = []
    for name, mod in (("onnx", "onnxruntime"), ("torch", "torch")):
        try:
            __import__(mod)
            out.append(name)
        except ImportError:
            pass
    return out


RUNTIMES = _runtimes()


def make_tts(runtime, model_dir, **kw):
    kw.setdefault("threads", int(os.environ.get("STEERTTS_EDGE_THREADS", "4")))
    if runtime == "onnx":
        from steertts_edge import OnnxTTS
        return OnnxTTS(model_dir, **kw)
    from steertts_edge import EdgeTTS
    return EdgeTTS(model_dir, **kw)


@pytest.fixture(scope="session", params=RUNTIMES)
def tts(request, model_dir):
    """Every functional test runs once per installed runtime (ONNX Runtime, PyTorch)."""
    if request.param == "onnx" and not (Path(model_dir) / "onnx" / "enc.onnx").exists():
        pytest.skip("no onnx/ in the package")
    return make_tts(request.param, model_dir)


def check_audio(y, min_s=0.2, max_s=30.0, min_peak=0.02):
    """Finite, non-silent float32 mono audio of a sane length."""
    assert isinstance(y, np.ndarray) and y.dtype == np.float32 and y.ndim == 1
    assert np.isfinite(y).all(), "non-finite samples"
    dur = len(y) / SR
    assert min_s <= dur <= max_s, f"duration {dur:.2f} s"
    peak = float(np.abs(y).max())
    assert min_peak < peak <= 1.5, f"peak {peak:.4f}"
    return dur
