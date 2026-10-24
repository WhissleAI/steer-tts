"""Request validation parity: the C library's error messages vs the Python packages', on invalid requests.

    cd c && PYTHONPATH=<repo>:<repo>/edge python tests/check_messages.py     (models in c/models, library in c/build)
"""
import json, os, sys
os.environ.setdefault("STEERTTS_LIB", os.path.abspath("build/libsteertts.so"))
sys.path.insert(0, "python")
from steertts_c import Engine, SteerTTSError
from steertts.request import normalize_request
from steertts import SteerTTS
v1py = SteerTTS("models/steer-tts", device="cpu", flow=False)
v1c = Engine("models/steer-tts", flow=False)
bad = [{"text": "", "pace": 9}, {"text": "x", "tag": "bogus", "ctrl": {"rate": 9}}, {"text": "x", "zz": 1, "aa": 2},
       {"text": "x", "voice": 5}, {"text": " "}, {"text": "x", "ctrl": {"rate": 9}}, {"text": "x", "ctrl": {"speed": 1}},
       {"text": "x", "ctrl": {"rate": True}}, {"text": "x", "ctrl": [1]}, {"text": "x", "pace": "fast"}, {"text": "x", "guidance": 5},
       {"text": "x", "flow": 1}, {"text": "x", "markup": "yes"}, {"text": "x", "seed": -1}, {"text": "x", "seed": 1.5},
       {"text": "x", "word_shift": [1]}, {"text": "x", "word_shift": {"a": {}}}, {"text": "x", "word_shift": {"-1": {}}},
       {"text": "x", "word_shift": {"1": {"pitch": 99}}}, {"text": "x", "word_shift": {"1": {"bogus": 1}}},
       {"text": "x", "word_shift": {"1": {"dur": 2}}, "markup": True}, {"text": "x", "voice": "voice_0005", "ref_wav": "a.wav"},
       {"text": "x", "prompt": 5}, {"text": "x", "tag": 5}, {"text": "x", "voice": "nobody"}, {"text": "x" * 5001}, {"text": 5}]
ok = 0
for r in bad:
    try: normalize_request(r, v1py); p = "OK"
    except ValueError as e: p = str(e)
    try: v1c.resolve(r); c = "OK"
    except SteerTTSError as e: c = str(e)
    same = p == c or (p.startswith("unknown tag") and c.startswith("unknown tag") and p[:40] == c[:40])
    ok += same
    if not same: print("V1 DIFF", json.dumps(r)[:80], "\n   py:", p[:150], "\n   c :", c[:150])
print(f"v1: {ok}/{len(bad)} identical")
from steertts_edge import EdgeTTS
epy = EdgeTTS("models/steer-tts-edge"); ec = Engine("models/steer-tts-edge")
ebad = [{"text": "x", "tag": ""}, {"text": "x", "voice": "nobody"}, {"text": "x", "tag": "bogus", "voice": "nobody"},
        {"text": "x", "word_shift": {"1": {"dur": 2}}, "markup": True}, {"text": "x", "ctrl": {"rate": 7}},
        {"text": "x", "ctrl": {"speed": 1}}, {"text": "x", "prompt": "sad"}, {"text": "x", "ref_wav": "a.wav"}, {"text": ""}]
ok = 0
for r in ebad:
    kw = {k: v for k, v in r.items() if k != "text"}
    try: y = epy(r["text"], **kw); p = f"OK {len(y)}"
    except ValueError as e: p = str(e)
    try: y = ec.synthesize(r); c = f"OK {len(y)}"
    except SteerTTSError as e: c = str(e)
    ok += p == c
    if p != c: print("EDGE DIFF", json.dumps(r)[:80], "\n   py:", p[:150], "\n   c :", c[:150])
print(f"edge: {ok}/{len(ebad)} identical")
sys.exit(0 if ok == len(ebad) else 1)
