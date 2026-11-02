#!/usr/bin/env python3
"""Convert the front end golden files to a flat TSV for test_front (no JSON parser in C).

    python golden_to_tsv.py ../../../native/tests/front ../../../tests/front_cases.json > golden.tsv

Lines (fields hex-encoded UTF-8 where they are text):
  P  text  norm|-  norm_error|-  ids|-  widx|-  error|-
  S  text  norm                                   (shared normalisation cases)
  G  word  phones                                 (GRU G2P cases, raw model indices)
  M  text  tag|-  ctrl|-  segs|-  error|-         (markup; ctrl: name=value,...; segs: JSON, hex)
"""
import json
import struct
import sys


def f32(v):
    return struct.pack("<f", float(v)).hex()


def canon_segs(segs):
    """segment list -> text~tag~ctrl~ws~pause~skip per segment, joined by '|' (floats as float32 bit patterns)"""
    out = []
    for s in segs:
        pause = s["pause_ms"] if isinstance(s["pause_ms"], int) else 0
        skip = int(pause == -1 or bool(s["event"]))
        ctrl = ",".join(f"{n}={f32(v)}" for n, v in s["ctrl"])
        ws = ",".join(f"{w[0]}/{f32(w[1])}/{f32(w[2])}/{f32(w[3])}" for w in s["ws"])
        out.append("~".join([s["text"].encode("utf-8").hex(), s["tag"], ctrl, ws, str(pause), str(skip)]))
    return "|".join(out)


def hx(s):
    return s.encode("utf-8").hex() if s is not None else "-"


def ints(v):
    return " ".join(str(x) for x in v) if v is not None else "-"


def main():
    gdir, shared = sys.argv[1], sys.argv[2]
    out = sys.stdout
    for line in open(f"{gdir}/golden.jsonl", encoding="utf-8"):
        d = json.loads(line)
        if d["k"] == "ph":
            out.write("\t".join(["P", hx(d["text"]), hx(d.get("norm")), hx(d.get("norm_error")),
                                 ints(d.get("ids")), ints(d.get("widx")), hx(d.get("error"))]) + "\n")
        else:
            ctrl = d.get("ctrl")
            cs = ",".join(f"{n}={f32(v)}" for n, v in ctrl) if ctrl is not None else None
            segs = canon_segs(d["segs"]) if "segs" in d else None
            out.write("\t".join(["M", hx(d["text"]), hx(d.get("tag")), hx(cs), hx(segs), hx(d.get("error"))]) + "\n")
    for d in json.load(open(shared, encoding="utf-8")):
        out.write("\t".join(["S", hx(d["text"]), hx(d["norm"])]) + "\n")
    for line in open(f"{gdir}/golden_g2p.jsonl", encoding="utf-8"):
        d = json.loads(line)
        out.write("\t".join(["G", hx(d["w"]), ints(d["p"])]) + "\n")


if __name__ == "__main__":
    main()
