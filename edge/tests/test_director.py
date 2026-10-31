"""The director (steertts_edge.director, numpy): plans against the golden plans of the C library's test
(c/tests/golden/director/plans.txt, from the trained model on held-out chapters), readings equal to whole-text
planning, and the C library itself through its ctypes binding when it is built. Needs director.bin in the model
directory (or in ../c/models/steer-tts-edge when testing from the steer-tts repo); skipped without it."""
import os
import sys
from pathlib import Path

import pytest

from conftest import MODEL
from steertts_edge.director import Director

REPO = Path(__file__).resolve().parents[2]
GOLDEN = REPO / "c/tests/golden/director/plans.txt"


def _director_path():
    for d in (Path(MODEL), REPO / "c/models/steer-tts-edge"):
        if (d / "director.bin").exists():
            return d
    pytest.skip("no director.bin")


@pytest.fixture(scope="module")
def director():
    return Director(_director_path())


def _chapters():
    genre, rows = None, []
    for line in GOLDEN.read_text().splitlines():
        if line.startswith("chapter "):
            if rows:
                yield genre, rows
            genre, rows = int(line[8:]), []
            continue
        head, text = line.split("\t", 1)
        f = head.split()
        rows.append(({"text": text, "dialogue": int(f[0]), "has_speaker": int(f[1]), "para_len": int(f[2]),
                      "para_index": int(f[3])}, [float(x) for x in f[4:10]]))
    if rows:
        yield genre, rows


def test_golden(director):
    if not GOLDEN.exists():
        pytest.skip("golden plans are in the steer-tts repo")
    worst, n = 0.0, 0
    for g, rows in _chapters():
        genre = Director.genres()[g] if g >= 0 else None
        got = director.plan([s for s, _ in rows], genre=genre)
        worst = max(worst, max(abs(a - b) for (_, w), p in zip(rows, got) for a, b in zip(w, p)))
        r = director.reading(genre)
        for i, (s, _) in enumerate(rows):
            assert r.next(s, rows[i + 1][0] if i + 1 < len(rows) else None) == got[i]
        n += len(rows)
    assert n > 500 and worst < 2e-3


def test_controls_and_pauses(director):
    p = director.plan(["It was a dark and stormy night.", "“Who goes there?” he cried.", "Nobody answered."])
    assert len(p) == 3 and all(len(x) == 6 for x in p)
    c = director.ctrl(p[1])
    assert set(c) == {"rate", "loud", "f0_med", "f0_iqr", "f0_end"} and all(abs(v) <= 3 for v in c.values())
    assert all(v == 0 for v in director.ctrl(p[1], strength=0).values())
    assert 80 <= Director.pause_ms(p[2]) <= 1500
    assert Director.pause_ms([0, 0, 0, 0, 0, 5]) == 1500 and Director.pause_ms([0, 0, 0, 0, 0, -5]) == 80


def test_same_as_c_library(director):
    lib = REPO / "c/build" / ("libsteertts.dylib" if sys.platform == "darwin" else "libsteertts.so")
    if not lib.exists():
        pytest.skip("the C library is not built")
    os.environ.setdefault("STEERTTS_LIB", str(lib))
    sys.path.insert(0, str(REPO / "c/python"))
    from steertts_c import Director as CDirector
    c = CDirector(str(_director_path()))
    texts = ["It is impossible to say how first the idea entered my brain; but once conceived, it haunted me day "
             "and night.", "Object there was none.", "Passion there was none.", "I loved the old man."]
    for a, b in zip(director.plan(texts, "gothic"), c.plan(texts, "gothic")):
        assert a == b
        assert director.controls(a) == c.controls(b) and Director.pause_ms(a) == CDirector.pause_ms(b)


def test_errors(tmp_path):
    with pytest.raises(OSError):
        Director(tmp_path)
    (tmp_path / "director.bin").write_bytes(b"XXXX" + bytes(64))
    with pytest.raises(ValueError):
        Director(tmp_path)
