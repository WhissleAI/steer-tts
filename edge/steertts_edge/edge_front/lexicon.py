"""Compact pronunciation lexicon: sorted words, front-coded in blocks, 1-byte phone codes, xz-compressed on disk.

Two column streams (words / phones) compress far better than interleaved entries (516 kB vs 620 kB for CMUdict).
Decompressed layout (little endian):
    b"LEX2" | u32 n_entries | u32 block | u32 n_blocks | u16 n_phones | n_phones x (u8 len, ascii phone)
    | u32 word_offs[n_blocks] | u32 phone_offs[n_blocks] | u32 len(word stream) | word stream | phone stream
word entry : u8 (128 + shared prefix length with the previous word; 0 at a block head) + ascii suffix (< 128)
phone entry: phone codes (1 + index into the phone table) + 0 terminator
Only the FIRST pronunciation of each CMUdict word is stored (g2p_en only ever uses cmu[w][0]); keys are exactly
nltk's cmudict.dict() keys, so membership tests (`w in lex`) match g2p_en's `w in self.cmu`.
"""
import bisect
import lzma
import struct
from functools import lru_cache


def build(entries, block=32):
    """entries: {word: [phones]} -> compressed bytes."""
    words = sorted(entries)
    phones = sorted({p for w in words for p in entries[w]})
    pid = {p: i + 1 for i, p in enumerate(phones)}
    assert len(phones) < 255
    W, P, wo, po, prev = bytearray(), bytearray(), [], [], ""
    for k, w in enumerate(words):
        assert w.isascii() and w and len(w) < 128
        c = 0
        if k % block == 0:
            wo.append(len(W))
            po.append(len(P))
        else:
            m = min(len(prev), len(w), 127)
            while c < m and prev[c] == w[c]:
                c += 1
        W += bytes([128 + c]) + w[c:].encode("ascii")
        P += bytes(pid[p] for p in entries[w]) + b"\0"
        prev = w
    head = b"LEX2" + struct.pack("<IIIH", len(words), block, len(wo), len(phones))
    head += b"".join(bytes([len(p)]) + p.encode() for p in phones)
    head += struct.pack(f"<{len(wo)}I", *wo) + struct.pack(f"<{len(po)}I", *po) + struct.pack("<I", len(W))
    return lzma.compress(bytes(head + W + P), preset=9 | lzma.PRESET_EXTREME)


class Lexicon:
    def __init__(self, path_or_bytes):
        raw = path_or_bytes if isinstance(path_or_bytes, (bytes, bytearray)) else open(path_or_bytes, "rb").read()
        b = lzma.decompress(raw) if raw else b"LEX2" + struct.pack("<IIIH", 0, 32, 0, 0) + struct.pack("<I", 0)
        assert b[:4] == b"LEX2"
        self.n, self.block, nb, nph = struct.unpack_from("<IIIH", b, 4)
        o = 18
        self.phones = [None]
        for _ in range(nph):
            L = b[o]
            self.phones.append(b[o + 1:o + 1 + L].decode())
            o += 1 + L
        self.wo = struct.unpack_from(f"<{nb}I", b, o)
        self.po = struct.unpack_from(f"<{nb}I", b, o + 4 * nb)
        (nw,) = struct.unpack_from("<I", b, o + 8 * nb)
        o += 8 * nb + 4
        self.W = b[o:o + nw].decode("latin-1")   # str: prefix bytes >= 128 become chars >= \x80
        self.P = b[o + nw:]
        self.heads = [self._words(k, 1)[0] for k in range(nb)]
        self.get = lru_cache(maxsize=8192)(self._get)

    def _block(self, k):
        we = self.wo[k + 1] if k + 1 < len(self.wo) else len(self.W)
        return self.W[self.wo[k]:we]

    def _words(self, k, limit=None):
        s, out, cur, i = self._block(k), [], "", 0
        n = len(s)
        while i < n and (limit is None or len(out) < limit):
            c = ord(s[i]) - 128
            j = i + 1
            while j < n and ord(s[j]) < 128:
                j += 1
            cur = cur[:c] + s[i + 1:j]
            out.append(cur)
            i = j
        return out

    def _phones(self, k, idx):
        P, p = self.P, self.po[k]
        for _ in range(idx):
            p = P.index(0, p) + 1
        e = P.index(0, p)
        return [self.phones[c] for c in P[p:e]]

    def _get(self, word):
        """-> phone list or None."""
        k = bisect.bisect_right(self.heads, word) - 1
        if k < 0:
            return None
        for idx, w in enumerate(self._words(k)):
            if w == word:
                return self._phones(k, idx)
            if w > word:
                return None
        return None

    def __contains__(self, word):
        return isinstance(word, str) and word.isascii() and self.get(word) is not None

    def __getitem__(self, word):
        r = self.get(word) if isinstance(word, str) and word.isascii() else None
        if r is None:
            raise KeyError(word)
        return r

    def __len__(self):
        return self.n

    def items(self):
        out = []
        for k in range(len(self.wo)):
            for idx, w in enumerate(self._words(k)):
                out.append((w, self._phones(k, idx)))
        return out
