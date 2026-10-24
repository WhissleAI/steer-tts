"""Release hygiene: payload size, file layout, config, voice names."""
import json
import re
from pathlib import Path

import pytest

BUDGET = 12_000_000          # PyTorch runtime payload (the 1.0.0 device budget)
BUDGET_ONNX = 12_700_000     # ONNX Runtime payload (same int4 / int8 codes; graph structure + shape annotations)
FRONT = ["front/cmudict.lex", "front/front.json.xz", "front/gru.npz", "front/oov_cache.lex"]
PAYLOAD = ["config.json", "acoustic.safetensors", "vocoder.safetensors", "voices.safetensors", "voices.json", "NOTICE",
           *FRONT]
PAYLOAD_ONNX = ["config.json", "onnx/enc.onnx", "onnx/dec.onnx", "onnx/voc.onnx", "voices.safetensors", "voices.json",
                "NOTICE", *FRONT]


def _payload(d, names=PAYLOAD):
    d = Path(d)
    files = [d / f for f in names]
    code = d / "steertts_edge"
    if code.is_dir():  # repo checkout (the installed package has its code in site-packages)
        files += [p for p in code.rglob("*.py")]
    return files


def test_payload_within_budget(model_dir):
    files = _payload(model_dir)
    assert all(f.is_file() and not f.is_symlink() for f in files), [str(f) for f in files if not f.is_file()]
    total = sum(f.stat().st_size for f in files)
    assert total <= BUDGET, f"{total:,} > {BUDGET:,}"


def test_onnx_payload_within_budget(model_dir):
    if not (Path(model_dir) / "onnx").is_dir():
        pytest.skip("no onnx/ in the package")
    files = _payload(model_dir, PAYLOAD_ONNX)
    assert all(f.is_file() and not f.is_symlink() for f in files), [str(f) for f in files if not f.is_file()]
    total = sum(f.stat().st_size for f in files)
    assert total <= BUDGET_ONNX, f"{total:,} > {BUDGET_ONNX:,}"


def test_no_symlinks(model_dir):
    links = [str(p) for p in Path(model_dir).rglob("*") if p.is_symlink()]
    assert not links, links


def test_config_clean(model_dir):
    cfg = json.load(open(Path(model_dir) / "config.json"))
    assert cfg["sample_rate"] == 24000 and cfg["acoustic"]["file"] == "acoustic.safetensors"
    assert "qat" not in cfg["acoustic"] and "src" not in cfg["vocoder"]
    v = cfg["vocoder"]
    assert (v["dim"], v["inter"], v["layers"]) == (384, 896, 8)


def test_voices_and_tags(tts):
    assert len(tts.voices) == 200 and len(set(tts.voices)) == 200
    assert all(re.fullmatch(r"voice_\d{4}", v) for v in tts.voices)
    assert tts.default_voice == "voice_1688" and "voice_1361" in tts.voices
    assert tts.tags[0] == "none" and len(tts.tags) == 51 and len(set(tts.tags)) == 51   # 50 style tags + none
    assert len(tts.controls) == 12


def test_voices_json_public_fields_only(model_dir):
    vj = json.load(open(Path(model_dir) / "voices.json"))
    keys = set().union(*(v.keys() for v in vj["voices"]))
    assert keys <= {"name", "gender", "pitch_register", "pace", "f0_median_hz"}, keys
