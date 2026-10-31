"""Runtime hygiene: no heavy front-end / vocoder packages, from_pretrained, CLI."""
import os
import subprocess
import sys

import numpy as np
import pytest

FORBIDDEN = ["nltk", "g2p_en", "vocos", "transformers", "torchaudio", "num2words", "inflect", "soundfile"]


@pytest.mark.parametrize("runtime", ["OnnxTTS", "EdgeTTS"])
def test_no_heavy_imports(model_dir, runtime):
    pytest.importorskip("onnxruntime" if runtime == "OnnxTTS" else "torch")
    forbidden = FORBIDDEN + (["torch"] if runtime == "OnnxTTS" else [])   # the ONNX runtime never imports torch
    code = (
        "import sys\n"
        f"from steertts_edge import {runtime} as R\n"
        f"t = R({model_dir!r})\n"
        "t('Numbers like 3,499 and Dr. Zyxquorpian.'); t('*Hi* [pause 200ms] <whisper>there</whisper>.', markup=True)\n"
        "list(t.stream('One. Two.'))\n"
        f"bad = [m for m in {forbidden!r} if m in sys.modules]\n"
        "print('FORBIDDEN', bad)\n"
        "print('FILE', __import__('steertts_edge').__file__)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert "FORBIDDEN []" in out, out
    if os.environ.get("STEERTTS_EDGE_REQUIRE_INSTALLED"):   # fresh-env run: code must come from site-packages
        assert "site-packages" in out.split("FILE", 1)[1], out


def test_from_pretrained_local_dir(model_dir, tts):
    # same thread count: ONNX Runtime splits reductions by thread count, so renders are deterministic per
    # (device, threads), and differ at ~1e-7 across thread counts
    t2 = type(tts).from_pretrained(model_dir, threads=int(os.environ.get("STEERTTS_EDGE_THREADS", "4")))
    assert np.array_equal(t2("Same weights."), tts("Same weights."))


def test_import_is_lazy():
    out = subprocess.run([sys.executable, "-c", "import sys, steertts_edge; print('torch' in sys.modules, "
                          "'onnxruntime' in sys.modules)"], capture_output=True, text=True, check=True).stdout
    assert out.split() == ["False", "False"], out


@pytest.mark.parametrize("runtime", ["onnx", "torch"])
def test_cli(model_dir, tmp_path, runtime):
    import wave
    pytest.importorskip("onnxruntime" if runtime == "onnx" else "torch")
    out = tmp_path / "cli.wav"
    exe = [sys.executable, "-m", "steertts_edge.cli", "--runtime", runtime]
    r = subprocess.run(exe + ["Hello from the command line.", "-o", str(out), "--model", model_dir,
                              "--voice", "voice_1361", "--tag", "happiness", "--ctrl", "f0_iqr=1", "--rate", "0.5"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    with wave.open(str(out)) as w:
        assert w.getframerate() == 24000 and w.getnframes() > 24000
    r = subprocess.run(exe + ["x", "--model", model_dir, "--tag", "angry", "-o", str(out)], capture_output=True, text=True)
    assert r.returncode == 2 and "unknown tag" in r.stderr
    r = subprocess.run(exe + ["--list", "tags", "--model", model_dir], capture_output=True, text=True, check=True)
    assert r.stdout.split()[:2] == ["none", "adoration"] and len(r.stdout.split()) == 51
