"""The Python binding's Director against the golden plans (golden/director/plans.txt), as test_director.c does for C:
whole-chapter plans within the same tolerance, and a reading fed one sentence at a time equal to them. Needs the
built library (../build) and director.bin in the edge model directory ($STEERTTS_EDGE_MODEL, default
../models/steer-tts-edge); skipped without it.   python3 test_director_py.py"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "python"))
from steertts_c import Director, SteerTTSError  # noqa: E402

TOL = 2e-3


def chapters():
    genre, rows = None, []
    for line in (HERE / "golden/director/plans.txt").read_text().splitlines():
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


def main():
    try:
        d = Director(os.environ.get("STEERTTS_EDGE_MODEL", str(HERE.parent / "models/steer-tts-edge")))
    except SteerTTSError as e:
        print(f"skip: {e}")
        return 0
    genres = Director.genres()
    worst, n = 0.0, 0
    for g, rows in chapters():
        genre = genres[g] if 0 <= g < len(genres) else None
        got = d.plan([s for s, _ in rows], genre=genre)
        worst = max(worst, max(abs(a - b) for (_, w), p in zip(rows, got) for a, b in zip(w, p)))
        r = d.reading(genre)
        for i, (s, _) in enumerate(rows):
            p = r.next(s, rows[i + 1][0] if i + 1 < len(rows) else None)
            assert max(abs(a - b) for a, b in zip(p, got[i])) < 1e-6, "reading differs from plan"
        n += len(rows)
    assert worst < TOL, f"plans differ from golden by {worst}"
    c = d.ctrl(got[0])
    assert set(c) == {"rate", "loud", "f0_med", "f0_iqr", "f0_end"} and all(abs(v) <= 3 for v in c.values())
    assert 80 <= Director.pause_ms(got[0]) <= 1500
    print(f"ok: {n} sentences, worst {worst:.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
