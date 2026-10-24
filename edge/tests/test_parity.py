"""Parity with the evaluated build: set STEERTTS_EDGE_REF to the source package directory (the reference
build the leaderboard numbers came from, with its own steertts_edge/ code). Renders the same requests with both
and compares sample by sample."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REF = os.environ.get("STEERTTS_EDGE_REF")
CASES = [
    dict(text="The meeting has been moved to Thursday afternoon at three."),
    dict(text="Are you coming with us tonight?", voice="voice_1361", ctrl={"f0_end": 2, "f0_iqr": 1.5}),
    dict(text="I really thought it would work this time.", tag="sadness", guidance=1.5),
    dict(text="Please keep this between the two of us.", tag="whisper", ctrl={"rate": -1.5}),
    dict(text="It costs $3,499.99 on March 3rd, Dr. Zyxquorpian.", voice="voice_0047"),
    dict(text="Well, *that* was fast. [pause 400ms] <slow>Let me think.</slow> <whisper>Shh.</whisper>", markup=True),
    dict(text="Word shift on one word here.", word_shift={"2": {"pitch": 3, "energy": 4, "dur": 1.3}}),
]

RENDER = r"""
import json, sys, numpy as np
sys.path.insert(0, sys.argv[1])
from steertts_edge import EdgeTTS
t = EdgeTTS(sys.argv[1], threads=4)
out = {}
for i, c in enumerate(json.loads(sys.argv[2])):
    c = dict(c)
    if "word_shift" in c:
        c["word_shift"] = {int(k): v for k, v in c["word_shift"].items()}
    out[str(i)] = t(c.pop("text"), **c)
np.savez(sys.argv[3], **out)
print(__import__("steertts_edge").__file__)
"""


@pytest.mark.skipif(not REF, reason="set STEERTTS_EDGE_REF to the source package dir")
def test_parity_with_source_package(tts, tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, "-c", RENDER, REF, json.dumps(CASES), str(tmp_path / "ref.npz")],
                       capture_output=True, text=True, env=env, cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert str(Path(REF).resolve()) in r.stdout, r.stdout     # the reference really ran the source package's code
    ref = np.load(tmp_path / "ref.npz")
    for i, c in enumerate(CASES):
        c = dict(c)
        if "word_shift" in c:
            c["word_shift"] = {int(k): v for k, v in c["word_shift"].items()}
        y = tts(c.pop("text"), **c)
        assert y.shape == ref[str(i)].shape, (i, y.shape, ref[str(i)].shape)
        assert np.array_equal(y, ref[str(i)]), (i, float(np.abs(y - ref[str(i)]).max()))
