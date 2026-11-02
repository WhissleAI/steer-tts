#!/usr/bin/env python3
"""Build front.bin, the flash-resident text front end data, from the edge front end assets.

    python tools/make_front_bin.py <steer-tts-edge>/native/front -o front.bin

Input: the native front directory of the SteerTTS edge model (lexicon.bin, oov.bin, gru.bin, hetero.tsv,
unicode.bin). Output: one read-only file that the device front end (components/steertts_nano/front) reads in place
from flash: sorted tables searched with binary search, a prefix-compressed block-indexed lexicon, the int8 G2P GRU.
Layout: docs/FRONT_END.md.
"""
import argparse
import struct
import sys

MAGIC = b"STFB"
MAGIC_GRU = b"STGR"
VERSION = 1
BLOCK = 32   # lexicon entries per index block

SEC_UCD_WORD, SEC_UCD_SPACE, SEC_UCD_DIGIT, SEC_UCD_G2P_ASCII, SEC_UCD_G2P, SEC_UCD_FOLD, SEC_UCD_LOWER = 10, 11, 12, 13, 14, 15, 16
SEC_LEX, SEC_OOV = 20, 21
SEC_GRU = 30
SEC_HET_W, SEC_HET_T, SEC_HET_H, SEC_HET_FLAGS = 40, 41, 42, 43

_PUNCT = list(",.?!;:-")
_ARPA = ["AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY", "F", "G", "HH", "IH", "IY",
         "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH"]
_VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
SYMBOLS = ["<pad>", "<sp>"] + _PUNCT + [p + s for p in _ARPA for s in (["0", "1", "2"] if p in _VOWELS else [""])]
SYM2ID = {s: i for i, s in enumerate(SYMBOLS)}


def u32(*v):
    return struct.pack(f"<{len(v)}I", *v)


def read_lex(path):
    b = open(path, "rb").read()
    if b[:4] != b"STLX" or struct.unpack_from("<I", b, 4)[0] != 1:
        sys.exit(f"{path}: not a lexicon")
    n, wb, pb = struct.unpack_from("<3I", b, 8)
    wo = struct.unpack_from(f"<{n + 1}I", b, 20)
    po = struct.unpack_from(f"<{n + 1}I", b, 20 + 4 * (n + 1))
    w0 = 20 + 8 * (n + 1)
    words, ph = b[w0:w0 + wb], b[w0 + wb:w0 + wb + pb]
    return [(words[wo[i]:wo[i + 1]], ph[po[i]:po[i + 1]]) for i in range(n)]


def lex_section(entries):
    keys = [w for w, _ in entries]
    assert keys == sorted(keys) and len(set(keys)) == len(keys), "lexicon keys must be sorted and unique"
    nb = (len(entries) + BLOCK - 1) // BLOCK
    data, offs = bytearray(), []
    for bi in range(nb):
        offs.append(len(data))
        prev = b""
        for w, p in entries[bi * BLOCK:(bi + 1) * BLOCK]:
            k = 0
            while k < min(len(w), len(prev), 255) and w[k] == prev[k]:
                k += 1
            suf = w[k:]
            assert len(suf) < 256 and len(p) < 256
            data += bytes([k, len(suf)]) + suf + bytes([len(p)]) + p
            prev = w
    offs.append(len(data))
    return u32(len(entries), nb, BLOCK) + u32(*offs) + bytes(data)


def ucd_sections(path):
    b = open(path, "rb").read()
    if b[:4] != b"STUC" or struct.unpack_from("<I", b, 4)[0] != 1:
        sys.exit(f"{path}: not unicode.bin")
    o = [8]

    def r32():
        v = struct.unpack_from("<I", b, o[0])[0]
        o[0] += 4
        return v

    def rstr():
        c = r32()
        n = b[o[0]]
        s = b[o[0] + 1:o[0] + 1 + n]
        o[0] += 1 + n
        return c, s

    sec = {}
    n = r32()
    sec[SEC_UCD_WORD] = u32(n) + b"".join(u32(r32(), r32()) for _ in range(n))
    n = r32()
    sp = [r32() for _ in range(n)]
    sp = sorted(c for c in sp if c >= 128)
    sec[SEC_UCD_SPACE] = u32(len(sp), *sp)
    n = r32()
    dg = [(r32(), r32(), r32()) for _ in range(n)]
    sec[SEC_UCD_DIGIT] = u32(n) + b"".join(u32(*t) for t in dg)
    n = r32()
    asc, non = {}, {}
    for _ in range(n):
        c, s = rstr()
        (asc if c < 128 else non)[c] = s   # dict: the later entry wins
    a = bytearray()
    for c in range(128):
        s = asc.get(c, b"")
        assert len(s) <= 3
        a += bytes([len(s)]) + s + b"\0" * (3 - len(s))
    sec[SEC_UCD_G2P_ASCII] = bytes(a)
    cps = sorted(non)
    pool, ent = bytearray(), bytearray()
    for c in cps:
        s = non[c]
        ent += struct.pack("<IHBB", c, len(pool), len(s), 0)
        pool += s
    sec[SEC_UCD_G2P] = u32(len(cps)) + bytes(ent) + bytes(pool)
    n = r32()
    fold = {}
    for _ in range(n):
        c, f = r32(), r32()
        fold.setdefault(c, f)   # linear search: the first entry wins
    sec[SEC_UCD_FOLD] = u32(len(fold)) + b"".join(u32(c, fold[c]) for c in sorted(fold))
    n = r32()
    low = {}
    for _ in range(n):
        c, s = rstr()
        low[c] = [ord(ch) for ch in s.decode("utf-8")]
    ent = bytearray()
    for c in sorted(low):
        v = low[c]
        assert len(v) <= 3
        ent += u32(c, len(v), *(v + [0] * (3 - len(v))))
    sec[SEC_UCD_LOWER] = u32(len(low)) + bytes(ent)
    return sec


def gru_section(path):
    """gru.bin re-laid out with every array 16-byte aligned (int8 rows feed the SIMD dot product).
    Header: "STGR", u32 version 2, H, NG, NP, then 12 u32 array offsets (from the section start):
    enc_x f32[NG*3H], dec_x f32[NP*3H], enc_q i8[3H*H], enc_s, enc_b f32[3H], dec_q i8[3H*H], dec_s, dec_b f32[3H],
    fc_q i8[NP*H], fc_s, fc_b f32[NP], phon2sym i32[NP]."""
    g = open(path, "rb").read()
    if g[:4] != b"STGR" or struct.unpack_from("<I", g, 4)[0] != 1:
        sys.exit("gru.bin: bad header")
    H, NG, NP = struct.unpack_from("<3I", g, 8)
    sizes = [4 * NG * 3 * H, 4 * NP * 3 * H, 3 * H * H, 4 * 3 * H, 4 * 3 * H, 3 * H * H, 4 * 3 * H, 4 * 3 * H,
             NP * H, 4 * NP, 4 * NP, 4 * NP]
    o, arrays = 20, []
    for n in sizes:
        arrays.append(g[o:o + n])
        o += n
    if o != len(g):
        sys.exit("gru.bin: unexpected size")
    hdr = MAGIC_GRU + u32(2, H, NG, NP)
    off = (len(hdr) + 4 * 12 + 15) // 16 * 16
    offs, body = [], bytearray()
    for a in arrays:
        body += b"\0" * ((-(off + len(body))) % 16)
        offs.append(off + len(body))
        body += a
    return hdr + u32(*offs) + b"\0" * (off - len(hdr) - 48) + bytes(body)


def strtab(keys):
    offs, pool = [], bytearray()
    for k in keys:
        offs.append(len(pool))
        pool += k + b"\0"
    offs.append(len(pool))
    return offs, bytes(pool)


def het_sections(path):
    W, T, H, has_td = {}, {}, {}, 1
    for line in open(path, "rb").read().split(b"\n"):
        if not line:
            continue
        f = line.split(b"\t")
        if f[0] == b"H" and len(f) == 5:
            p1 = bytes(SYM2ID[s.decode()] for s in f[2].split())
            p2 = bytes(SYM2ID[s.decode()] for s in f[3].split())
            H[f[1]] = (p1, p2, f[4])
        elif f[0] == b"W" and len(f) == 3:
            W[f[1]] = int(f[2])
        elif f[0] == b"T" and len(f) == 3:
            T[f[1]] = f[2]
        elif f[0] == b"N":
            has_td = 0
        else:
            sys.exit(f"bad hetero.tsv line {line[:60]!r}")
    sec = {}
    wk = sorted(W)
    offs, pool = strtab(wk)
    sec[SEC_HET_W] = u32(len(wk)) + u32(*offs) + struct.pack(f"<{len(wk)}i", *[W[k] for k in wk]) + pool
    tk = sorted(T)
    ko, kp = strtab(tk)
    vo, vp = strtab([T[k] for k in tk])
    sec[SEC_HET_T] = u32(len(tk)) + u32(*ko) + u32(*vo) + u32(len(kp)) + kp + vp
    hk = sorted(H)
    pool, offs = bytearray(), []
    for k in hk:
        p1, p2, pos = H[k]
        offs.append(len(pool))
        pool += k + b"\0" + pos + b"\0" + bytes([len(p1)]) + p1 + bytes([len(p2)]) + p2
    offs.append(len(pool))
    sec[SEC_HET_H] = u32(len(hk)) + u32(*offs) + bytes(pool)
    sec[SEC_HET_FLAGS] = u32(has_td)
    return sec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("front_dir")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    d = a.front_dir.rstrip("/")
    sec = {}
    sec.update(ucd_sections(d + "/unicode.bin"))
    sec[SEC_LEX] = lex_section(read_lex(d + "/lexicon.bin"))
    sec[SEC_OOV] = lex_section(read_lex(d + "/oov.bin"))
    sec[SEC_GRU] = gru_section(d + "/gru.bin")
    sec.update(het_sections(d + "/hetero.tsv"))
    ids = sorted(sec)
    off = 12 + 12 * len(ids)
    off = (off + 15) // 16 * 16
    table, body = bytearray(), bytearray()
    for i in ids:
        pad = (-(off + len(body))) % 16
        body += b"\0" * pad
        table += u32(i, off + len(body), len(sec[i]))
        body += sec[i]
    hdr = MAGIC + u32(VERSION, len(ids)) + bytes(table)
    hdr += b"\0" * (off - len(hdr))
    out = hdr + bytes(body)
    open(a.out, "wb").write(out)
    sizes = {k: len(v) for k, v in sec.items()}
    print(f"{a.out}: {len(out)} bytes (lexicon {sizes[SEC_LEX]}, oov {sizes[SEC_OOV]}, gru {sizes[SEC_GRU]}, "
          f"hetero {sizes[SEC_HET_W] + sizes[SEC_HET_T] + sizes[SEC_HET_H]}, unicode "
          f"{sum(sizes[k] for k in sizes if 10 <= k < 20)})")


if __name__ == "__main__":
    main()
