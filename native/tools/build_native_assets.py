#!/usr/bin/env python3
"""Convert the SteerTTS edge front-end assets (pkg/front: cmudict.lex, oov_cache.lex, gru.npz, front.json.xz) into
the plain little-endian files the C++ front end (native/src/front) loads, plus the Unicode tables Python's `re` /
`str` / `unicodedata` semantics need.

usage: build_native_assets.py <pkg_dir> <out_dir> [--no-precompute]   (pkg_dir contains steertts_edge/ and front/)

Output files (all integers little endian, u32 unless noted):

lexicon.bin / oov.bin   residual CMUdict / OOV cache (first pronunciation of each word, keys sorted bytewise)
    "STLX" | u32 version=1 | u32 n | u32 words_bytes | u32 phones_bytes
    | u32 word_off[n+1] | u32 phone_off[n+1] | words blob (ascii, concatenated) | phones blob (u8 SYMBOLS ids)
    An entry with phone_off[i] == phone_off[i+1] has no phones: for lexicon.bin that is a "member-only" CMUdict word
    (pronunciation = GRU prediction), for oov.bin a cached empty prediction (returned as-is, like Python).
    By default the member-only CMUdict entries are filled here with the Python numpy GRU's own prediction (what
    EdgeG2p computes at runtime for them, mapped to SYMBOLS ids and with the tokens to_phonemes drops removed), so
    the C++ front end needs no GRU run for them: identical output by construction, ~+0.2 MB.
    --no-precompute keeps them phone-less (C++ then runs its GRU port, like Python).

gru.bin                 tier-B int8 GRU G2P (g2p_en's GRU, gru_np.GruG2P)
    "STGR" | u32 version=1 | u32 H(256) | u32 n_graph(29) | u32 n_phon(74)
    | f32 enc_x[n_graph][3H]   (numpy's own float32 enc_emb @ enc_w_ih.T + enc_b_ih, bit-exact copy)
    | f32 dec_x[n_phon][3H]    (same for the decoder)
    | i8 enc_hh[3H][H] | f32 enc_hh_scale[3H] | f32 enc_b_hh[3H]      (W_hh row j = q[j][:] * scale[j], exact)
    | i8 dec_hh[3H][H] | f32 dec_hh_scale[3H] | f32 dec_b_hh[3H]
    | i8 fc[n_phon][H] | f32 fc_scale[n_phon] | f32 fc_b[n_phon]
    | i32 phon2sym[n_phon]     (GRU phoneme index -> SYMBOLS id, -1 = dropped by to_phonemes: <pad> <unk> <s> </s> UW)

hetero.tsv              heteronym classifier + homograph table (text, one record per line, tab separated)
    H <word> <pron1 phones space-separated> <pron2> <pos1>
    W <feature key> <int weight>
    T <word> <coarse tag>

unicode.bin             Python 3 (unicodedata 14.0 for the generating interpreter) character tables
    "STUC" | u32 version=1
    | u32 n | n x (u32 lo, u32 hi)            re `\\w` (str patterns) as inclusive ranges
    | u32 n | n x u32 cp                      re `\\s` == str.isspace() == str.strip() whitespace
    | u32 n | n x (u32 lo, u32 hi, u32 v0)    re `\\d` ranges; int()/float() digit value of cp = v0 + (cp - lo)
    | u32 n | n x (u32 cp, u8 len, bytes)     g2p clean map: cp -> ascii string that
                                              re.sub("[^ a-z'.,?!\\-]", "", "".join(non-Mn chars of NFD(c)).lower())
                                              leaves (only non-empty results are stored)
    | u32 n | n x (u32 cp, u32 ascii)         re.IGNORECASE: non-ascii cp that matches ascii letter (lower case)
    | u32 n | n x (u32 cp, u8 len, bytes)     str.lower() (utf-8) of those non-ascii cps
"""
import json
import lzma
import re
import struct
import sys
import unicodedata
from pathlib import Path

import numpy as np


def main():
    pkg, out = Path(sys.argv[1]), Path(sys.argv[2])
    sys.path.insert(0, str(pkg))
    from steertts_edge.text import SYMBOLS, SYM2ID
    from steertts_edge.edge_front.lexicon import Lexicon
    from steertts_edge.edge_front.gru_np import GruG2P, PHONEMES
    front = pkg / "front"
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((pkg / "config.json").read_text())
    assert cfg["front"]["tier"] == "B" and (front / "gru.npz").exists()

    # ---------------------------------------------------------------- lexicons
    precompute = "--no-precompute" not in sys.argv
    gru = GruG2P(front / "gru.npz")

    def write_lex(src, dst, fill=False):
        raw = src.read_bytes() if src.exists() else b""
        items = Lexicon(raw).items() if raw else []
        if fill:  # member-only entries -> the numpy GRU prediction (EdgeG2p: cmu[w][0] = model.member(w))
            items = [(w, ph if ph else [p for p in gru.predict(w) if p in SYM2ID and p not in ("<pad>", "<sp>")])
                     for w, ph in items]
            assert all(ph for _, ph in items), "an empty member prediction would be indistinguishable"
        words = [w for w, _ in items]
        assert words == sorted(words, key=lambda w: w.encode()) and len(set(words)) == len(words)
        wb, pb, wo, po = bytearray(), bytearray(), [0], [0]
        for w, ph in items:
            assert w.isascii()
            wb += w.encode("ascii")
            pb += bytes(SYM2ID[p] for p in ph)
            wo.append(len(wb)); po.append(len(pb))
        n = len(items)
        data = b"STLX" + struct.pack("<IIII", 1, n, len(wb), len(pb))
        data += struct.pack(f"<{n + 1}I", *wo) + struct.pack(f"<{n + 1}I", *po) + bytes(wb) + bytes(pb)
        (out / dst).write_bytes(data)
        return n

    n_lex = write_lex(front / "cmudict.lex", "lexicon.bin", fill=precompute)
    n_oov = write_lex(front / "oov_cache.lex", "oov.bin")

    # ---------------------------------------------------------------- GRU
    g = gru
    z = np.load(front / "gru.npz")
    H = g.enc_hh.shape[0]
    assert g.enc_x.dtype == np.float32 and g.enc_x.shape == (29, 3 * H) and g.dec_x.shape == (74, 3 * H)

    def q(name):
        u, s = z[name + ".u"], z[name + ".s"].astype(np.float32)
        # the numpy path dequantises as u.astype(f32) * s (exact in f32: 8-bit x 11-bit mantissas)
        assert np.array_equal(u.astype(np.float32) * s[:, None], g.W[name])
        return u.astype(np.int8).tobytes() + s.astype("<f4").tobytes()

    def f32(a):
        return np.ascontiguousarray(a, dtype="<f4").tobytes()

    phon2sym = [SYM2ID[p] if (p in SYM2ID and p not in ("<pad>", "<sp>")) else -1 for p in PHONEMES]
    data = b"STGR" + struct.pack("<IIII", 1, H, 29, 74)
    data += f32(g.enc_x) + f32(g.dec_x)
    data += q("enc_w_hh") + f32(g.W["enc_b_hh"])
    data += q("dec_w_hh") + f32(g.W["dec_b_hh"])
    data += q("fc_w") + f32(g.W["fc_b"])
    data += struct.pack("<74i", *phon2sym)
    (out / "gru.bin").write_bytes(data)

    # ---------------------------------------------------------------- heteronyms
    meta = json.loads(lzma.decompress((front / "front.json.xz").read_bytes()))
    assert not meta.get("postag"), "PosHetero port not implemented (asset has no postag model)"
    het = meta["hetero"]
    lines = []
    for w, (a, b, pos) in meta["homographs"].items():
        for p in a.split() + b.split():
            assert p in SYM2ID, p
        lines.append(f"H\t{w}\t{a}\t{b}\t{pos}")
    for k, v in het["w"].items():
        assert "\t" not in k and "\n" not in k and isinstance(v, int)
        lines.append(f"W\t{k}\t{v}")
    for k, v in (het.get("tags") or {}).items():
        lines.append(f"T\t{k}\t{v}")
    if het.get("tags") is None:
        lines.append("N\tnotags")
    (out / "hetero.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---------------------------------------------------------------- unicode
    cps = [c for c in range(0x110000) if not (0xD800 <= c < 0xE000)]

    def ranges(pred):
        rs, lo, prev = [], None, None
        for c in cps:
            if pred(c):
                if lo is None:
                    lo = c
                elif c != prev + 1:
                    rs.append((lo, prev)); lo = c
                prev = c
        if lo is not None:
            rs.append((lo, prev))
        return rs

    rw, rs_, rd = re.compile(r"\w"), re.compile(r"\s"), re.compile(r"\d")
    word = ranges(lambda c: rw.match(chr(c)) is not None)
    space = [c for c in cps if rs_.match(chr(c))]
    assert space == [c for c in cps if chr(c).isspace()] == [c for c in cps if chr(c).strip() == ""]
    digits = [c for c in cps if rd.match(chr(c))]
    drs = []
    for c in digits:
        v = unicodedata.decimal(chr(c))
        assert int(chr(c)) == v and float(chr(c)) == v
        if drs and drs[-1][1] == c - 1 and drs[-1][2] + (c - drs[-1][0]) == v:
            drs[-1][1] = c
        else:
            drs.append([c, c, v])
    keep = re.compile("[^ a-z'.,?!\\-]")
    g2pmap = []
    for c in cps:
        s = "".join(ch for ch in unicodedata.normalize("NFD", chr(c)) if unicodedata.category(ch) != "Mn")
        s = keep.sub("", s.lower())
        if s:
            g2pmap.append((c, s.encode("ascii")))
    fold, lower = [], []
    for c in cps:
        if c < 128:
            continue
        hits = [x for x in "abcdefghijklmnopqrstuvwxyz" if re.fullmatch("(?i)" + x, chr(c))]
        if hits:
            assert len(hits) == 1
            fold.append((c, ord(hits[0])))
            lower.append((c, chr(c).lower().encode("utf-8")))
    # the C++ engine implements IGNORECASE as: fold(ch) (ascii lower or the table above) compared on both sides;
    # check that model against sre for every construct markup._TOKEN uses
    for c in cps:
        ch = chr(c)
        f = dict(fold).get(c, ord(ch.lower()) if c < 128 else c)
        exp = 97 <= f <= 122 or ch == "_"
        assert (re.fullmatch("(?i)[A-Za-z_]", ch) is not None) == exp, hex(c)
    b = bytearray(b"STUC" + struct.pack("<I", 1))
    b += struct.pack("<I", len(word)) + b"".join(struct.pack("<II", *r) for r in word)
    b += struct.pack("<I", len(space)) + b"".join(struct.pack("<I", c) for c in space)
    b += struct.pack("<I", len(drs)) + b"".join(struct.pack("<III", *r) for r in drs)
    b += struct.pack("<I", len(g2pmap)) + b"".join(struct.pack("<IB", c, len(s)) + s for c, s in g2pmap)
    b += struct.pack("<I", len(fold)) + b"".join(struct.pack("<II", c, a) for c, a in fold)
    b += struct.pack("<I", len(lower)) + b"".join(struct.pack("<IB", c, len(s)) + s for c, s in lower)
    (out / "unicode.bin").write_bytes(bytes(b))

    (out / "SYMBOLS.txt").write_text("\n".join(SYMBOLS) + "\n")
    print(f"lexicon {n_lex}, oov {n_oov}, gru H={H}, homographs {len(meta['homographs'])}, hetero w {len(het['w'])}, "
          f"word ranges {len(word)}, space {len(space)}, digit ranges {len(drs)}, g2pmap {len(g2pmap)}, "
          f"fold {len(fold)} -> {out}")


if __name__ == "__main__":
    main()
