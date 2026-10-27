"""The director (docs/director.md) in numpy: plans how each sentence of a longer text is performed (rate, loudness,
pitch level and range, the ending, the pause after) from the text and the sentences around it. A port of the C
library's c/src/st_director.c with the same API as its ctypes binding (c/python/steertts_c.py: Director):

    from steertts_edge.director import Director
    d = Director("steer-tts-edge")                  # the edge directory, or director.bin itself
    plans = d.plan(["“Who are you?” she asked.", "Nobody answered."], genre="children")
    ctrl, gap = d.ctrl(plans[0]), Director.pause_ms(plans[0])

The same tokenizer (CRC32 of "w:word" and "b:word_word" into 16384 buckets), the same 18 features, an embedding bag
(mean) -> tanh(Linear) sentence encoder, a GRU cell over the reading that sees the next sentence, its own previous
plan and (format 2) the mean of the last few sentence encodings, and a linear head. Weights are 8-bit with a float
scale per row. The arithmetic is float32 in the C's order (sums left to right, no fused multiply-add: the library
builds with -ffp-contract=off) and expf / tanhf from the platform's libm, so plans match the C library to the bit.
"""
import re
import struct
import zlib
from pathlib import Path

import numpy as np

PLAN_DIMS = 6
FEATURES = 18
GENRES = ("children", "adventure", "fantasy", "mystery", "scifi", "gothic", "literary", "humour", "nonfiction", "poetry")
CONTROLS = ("rate", "loud", "f0_med", "f0_iqr", "f0_end")
BUCKETS = 16384    # the tokenizer's, fixed (as the C: not read from the file)
MAX_TOKENS = 512   # words per sentence, and tokens (words, then word pairs)
END_TO_F0_END = np.float32(0.8)   # how the ending (the final word's pitch, z) maps to the f0_end control (end slope, z)
_QUOTES = ("\"", "“", "”", "‘")   # a str sentence holding one of these is dialogue (as steertts_c)

f32 = np.float32


# ---------------------------------------------------------------- loading

def _mat(buf, off, rows, cols):
    """An 8-bit matrix: rows float32 scales, then rows x cols int8. -> (q float32 [rows, cols], scale [rows]), end."""
    n_s, n_q = 4 * rows, rows * cols
    if off + n_s + n_q > len(buf):
        raise ValueError("director model file is truncated")
    scale = np.frombuffer(buf, "<f4", rows, off).astype(f32)
    q = np.frombuffer(buf, np.int8, n_q, off + n_s).reshape(rows, cols).astype(f32)
    return (q, scale), off + n_s + n_q


def _vec(buf, off, n):
    if off + 4 * n > len(buf):
        raise ValueError("director model file is truncated")
    return np.frombuffer(buf, "<f4", n, off).astype(f32), off + 4 * n


def _dot(m, x):
    """Every row of an 8-bit matrix dotted with x, as the C's dot_row: products summed left to right in float32,
    then times the row's scale. (cumsum accumulates in order; a plain dot would sum pairwise.)"""
    q, scale = m
    return np.cumsum(q * x, axis=1, dtype=f32)[:, -1] * scale


def _libm():
    """The C library's expf / tanhf come from the platform's libm, which is not correctly rounded (on macOS about 1%
    of tanhf results are an ulp off): calling the same functions keeps the port bit-exact with the C library built
    on this machine. Without a libm (e.g. Windows), float64 results rounded to float32 are within an ulp."""
    try:
        import ctypes
        import ctypes.util
        m = ctypes.CDLL(ctypes.util.find_library("m") or "libm.so.6")
        for f in (m.expf, m.tanhf):
            f.restype, f.argtypes = ctypes.c_float, [ctypes.c_float]
        return m.expf, m.tanhf
    except (OSError, AttributeError):
        return None


_LIBM = _libm()


def _exp(x):
    if _LIBM:
        return np.array([_LIBM[0](v) for v in x.tolist()], f32)
    return np.exp(x.astype(np.float64)).astype(f32)


def _tanh(x):
    if _LIBM:
        return np.array([_LIBM[1](v) for v in x.tolist()], f32)
    return np.tanh(x.astype(np.float64)).astype(f32)


def _sigm(x):
    return f32(1) / (f32(1) + _exp(-x))


# ---------------------------------------------------------------- tokens and features (as train_director.py)

_WORD = re.compile(rb"[a-z']+|[0-9]+|[!?]|\xe2\x80[\xa6\x94]")   # on the ASCII-lower-cased bytes; others separate


def _bytes(text):
    b = text.encode() if isinstance(text, str) else bytes(text)
    z = b.find(b"\0")
    return b if z < 0 else b[:z]   # the C reads a NUL-terminated string


def words(text):
    """The words of a text: [a-z']+ | [0-9]+ | ! ? … — (lower-cased ASCII, as bytes; each kept to 63 bytes)."""
    return [m.group()[:63] for m in _WORD.finditer(_bytes(text).lower())][:MAX_TOKENS]


def tokens(text):
    """The hashed tokens of a text, as training: CRC32 of "w:word" then "b:word_word" mod 16384, at most 512."""
    w = words(text)
    out = [zlib.crc32(b"w:" + x) % BUCKETS for x in w]
    for a, b in zip(w, w[1:]):
        if len(out) >= MAX_TOKENS:
            break
        out.append(zlib.crc32(b"b:" + a + b"_" + b) % BUCKETS)
    return out


_SPACE = frozenset(b" \t\n\v\f\r")   # isspace() in the C locale
_HEADS = (b"chapter", b"book", b"part", b"canto", b"stanza")


def _alnum_(c):
    return c < 128 and (chr(c).isalnum() or c == 95)   # isalnum() in the C locale, or '_'


def features(text, dialogue=0, has_speaker=0, para_len=1, para_index=0):
    """The sentence's 18 features (float32), as st_director_features."""
    t = _bytes(text)
    L = len(t)
    e = L   # the last character after trailing quotes, apostrophes, spaces, brackets and ” ’
    while True:
        if e >= 1 and t[e - 1] in b"\"' )":
            e -= 1
        elif e >= 3 and t[e - 3] == 0xE2 and t[e - 2] == 0x80 and t[e - 1] in (0x9D, 0x99):
            e -= 3
        else:
            break
    last = t[e - 1] if e >= 1 else 0
    ellipsis = e >= 3 and t[e - 3:e] == b"\xe2\x80\xa6"
    r = L
    while r > 0 and t[r - 1] in _SPACE:
        r -= 1
    if r >= 3 and t[r - 3:r] == b"...":
        ellipsis = True
    nwords, in_word = 0, False
    for c in t:
        sp = c in _SPACE
        if not sp and not in_word:
            nwords += 1
        in_word = not sp
    upper = any(65 <= c <= 90 for c in t)
    lower = any(97 <= c <= 122 for c in t)
    head = t[:15].lower() + bytes(16 - min(L, 15))
    titled = any(head.startswith(h) and not _alnum_(head[len(h)]) for h in _HEADS)   # "chapter", not "chapters"
    pos = f32(para_index) / f32(para_len - 1) if para_len > 1 else f32(0)
    return np.array([
        f32(min(nwords, 60)) / f32(20), f32(t.count(b",")) / f32(3), f32(t.count(b";") + t.count(b":")) / f32(2),
        last == 0x3F, last == 0x21, last == 0x2E, ellipsis,
        dialogue != 0, has_speaker != 0, b"\xe2\x80\x9c" in t or b"\"" in t,
        b"\xe2\x80\x94" in t or b"--" in t, upper and not lower and nwords > 1,
        para_index == 0, para_index == para_len - 1, f32(min(para_len, 12)) / f32(6), pos,
        nwords <= 4, titled], f32)


def _sentence(x, i, n):
    """str or dict -> (text, dialogue, has_speaker, para_len, para_index) (a str is one paragraph of n sentences;
    quotes mark dialogue), as steertts_c._sentence."""
    if isinstance(x, str):
        x = {"text": x}
    text = x["text"]
    dialogue = x.get("dialogue", any(q in text for q in _QUOTES))
    return (text, int(dialogue), int(x.get("has_speaker", 0)), int(x.get("para_len", n)), int(x.get("para_index", i)))


# ---------------------------------------------------------------- the model

class Director:
    """The director: plans how each sentence of a longer text is performed. A plan is 6 floats (rate, loud, f0_med,
    f0_iqr, end, gap); ctrl(plan) turns it into request controls for the paired voice, pause_ms(plan) into the
    silence after. path: director.bin, or a directory holding it (the edge model's)."""

    GENRES = GENRES

    def __init__(self, path="steer-tts-edge"):
        p = Path(path)
        if p.suffix != ".bin":
            p = p / "director.bin"
        try:
            buf = p.read_bytes()
        except OSError as e:
            raise OSError(f"cannot open {p}") from e
        self._load(buf)

    @classmethod
    def from_bytes(cls, data):
        """The same from bytes in memory (steertts_director_from_memory)."""
        d = cls.__new__(cls)
        d._load(bytes(data))
        return d

    def _load(self, buf):
        if len(buf) < 32 or buf[:4] != b"STDR":
            raise ValueError("not a director model (STDR)")
        h = list(struct.unpack_from("<7I", buf, 4)) + [0]
        off = 32
        if h[0] == 2:   # version 2: scene memory
            h[7] = struct.unpack_from("<I", buf, off)[0] if len(buf) >= 36 else 0
            off += 4
        if h[0] not in (1, 2):
            raise ValueError("unsupported director model version")
        self.version = h[0]
        buckets, self.emb, self.hid, nfeat, ngenre, ntarget, self.scene = h[1:]
        if nfeat != FEATURES or ngenre != len(GENRES) or ntarget != PLAN_DIMS:
            raise ValueError("director model shape does not match this build")
        H, E = self.hid, self.emb
        if self.scene > 64 or H > 64 or E > 64:
            raise ValueError("director model shape is out of range")
        if buckets < BUCKETS:   # the tokens index 16384 buckets whatever the file says (the C would read past it)
            raise ValueError("director model has fewer than 16384 buckets")
        in_enc, in_cell = E + nfeat + ngenre, (3 if self.scene else 2) * H + ntarget
        (q, s), off = _mat(buf, off, buckets, E)
        self._emb = q * s[:, None]   # the C's (float)q * scale, the same float32 product every time it is read
        self._enc, off = _mat(buf, off, H, in_enc)
        self._enc_b, off = _vec(buf, off, H)
        self._wih, off = _mat(buf, off, 3 * H, in_cell)
        self._whh, off = _mat(buf, off, 3 * H, H)
        self._bih, off = _vec(buf, off, 3 * H)
        self._bhh, off = _vec(buf, off, 3 * H)
        self._head, off = _mat(buf, off, ntarget, 2 * H)
        self._head_b, off = _vec(buf, off, ntarget)
        # optional: the paired voice's calibration ("CAL1" + 25 floats): request = want x cal
        self.cal = None
        if len(buf) - off >= 4 + 100 and buf[off:off + 4] == b"CAL1":
            self.cal = np.frombuffer(buf, "<f4", 25, off + 4).astype(f32).reshape(5, 5)

    def _encode(self, text, feat, genre):
        """tanh(enc(mean(emb[tokens]) ++ features ++ genre)) -> s[hid]"""
        tok = tokens(text) or [0]   # as training: an empty bag is bucket 0
        x = np.cumsum(self._emb[tok], axis=0, dtype=f32)[-1] / f32(len(tok))
        return _tanh(_dot(self._enc, np.concatenate([x, feat, genre])) + self._enc_b)

    @staticmethod
    def genres():
        return list(GENRES)

    def plan(self, sentences, genre=None):
        """sentences: strs (one paragraph) or dicts {text, dialogue, has_speaker, para_len, para_index}, in order.
        Returns one plan (a list of 6 floats: rate, loud, f0_med, f0_iqr, end, gap) per sentence."""
        n = len(sentences)
        ss = [_sentence(x, i, n) for i, x in enumerate(sentences)]
        r = _Reading(self, genre)
        return [r._next(s, ss[i + 1] if i + 1 < n else None) for i, s in enumerate(ss)]

    def controls(self, plan, strength=1.0):
        """The plan as control z-scores for the paired voice: {rate, loud, f0_med, f0_iqr, f0_end}, calibrated with
        the file's CAL1 block when it has one, clamped to ±3."""
        p, k = np.asarray(plan, f32), f32(strength)
        want = np.array([k * p[0], k * p[1], k * p[2], k * p[3], k * END_TO_F0_END * p[4]], f32)
        out = want
        if self.cal is not None:   # the voice delivers only part of a requested change (pitch least): ask for more
            out = np.cumsum(want[:, None] * self.cal, axis=0, dtype=f32)[-1]
        return dict(zip(CONTROLS, (float(v) for v in np.clip(out, -3, 3))))

    def ctrl(self, plan, strength=1.0):
        """The same, as a request's ctrl dict (pass as tts(text, ctrl=...))."""
        return self.controls(plan, strength)

    @staticmethod
    def pause_ms(plan, pace=1.0):
        """Silence to leave after the sentence (80 .. 1500 ms), shorter at a faster pace."""
        pace = f32(pace)
        e = _exp(np.array([plan[5]], f32))[0]
        ms = (e - f32(1)) * f32(1000) / (pace if pace > f32(0.25) else f32(1))
        return 80 if ms < 80 else 1500 if ms > 1500 else int(ms + f32(0.5))

    def reading(self, genre=None):
        """Sentence by sentence (messages as they arrive), keeping the context: r = d.reading(); r.next(text)."""
        return _Reading(self, genre)


class _Reading:
    """The state of a reading: the GRU's hidden state, the previous plan, the scene memory (a ring of the last
    `scene` sentence encodings) and the last look-ahead sentence's encoding, reused when it becomes the current one
    (by its text alone, as the C)."""

    def __init__(self, d, genre=None):
        self._d = d
        self._genre = np.array([g == genre for g in GENRES], f32)
        self._h = np.zeros(d.hid, f32)
        self._prev = np.zeros(PLAN_DIMS, f32)
        self._past = np.zeros((d.scene, d.hid), f32)
        self._npast = self._next_past = 0
        self._ahead = None   # (text bytes, encoding)

    def next(self, sentence, following=None):
        """The plan for this sentence (6 floats); following: the sentence after it when known."""
        return self._next(_sentence(sentence, 0, 1), _sentence(following, 1, 2) if following is not None else None)

    def _next(self, s, nxt):
        d, H = self._d, self._d.hid
        text = _bytes(s[0])
        if self._ahead is not None and self._ahead[0] == text:
            enc = self._ahead[1]
        else:
            enc = d._encode(text, features(text, *s[1:]), self._genre)
        self._ahead = None
        if nxt is not None:
            nt = _bytes(nxt[0])
            nx = d._encode(nt, features(nt, *nxt[1:]), self._genre)
            self._ahead = (nt, nx)
        else:
            nx = np.zeros(H, f32)
        parts = [enc, nx]
        if d.scene:   # the mean of the previous sentences' encodings (up to `scene` of them; zeros at the start)
            if self._npast:
                parts.append(np.cumsum(self._past[:self._npast], axis=0, dtype=f32)[-1] / f32(self._npast))
            else:
                parts.append(np.zeros(H, f32))
        x = np.concatenate(parts + [self._prev])
        # GRUCell, PyTorch gate order r, z, n
        gi = _dot(d._wih, x) + d._bih
        gh = _dot(d._whh, self._h) + d._bhh
        rg = _sigm(gi[:H] + gh[:H])
        zg = _sigm(gi[H:2 * H] + gh[H:2 * H])
        ng = _tanh(gi[2 * H:] + rg * gh[2 * H:])
        self._h = (f32(1) - zg) * ng + zg * self._h
        plan = _dot(d._head, np.concatenate([self._h, enc])) + d._head_b
        self._prev = plan
        if d.scene:   # this sentence joins the scene memory
            self._past[self._next_past] = enc
            self._next_past = (self._next_past + 1) % d.scene
            self._npast = min(self._npast + 1, d.scene)
        return [float(v) for v in plan]
