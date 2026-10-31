"""Generate src/st_unicode_tab.c: the Unicode tables of the MiniLM WordPiece tokenizer (src/st_wordpiece.c).

    python tools/gen_unicode.py [out.c]

The reference is the tokenizer the model actually runs with: transformers' BertTokenizer, which (transformers >= 5)
is the Rust `tokenizers` BertNormalizer(clean_text, handle_chinese_chars, strip_accents=lowercase=True) +
BertPreTokenizer. Its Unicode data does not match any single Python unicodedata version (the normalizer's Mn / Cc / Cf
tables, its NFD tables and Rust's char::to_lowercase come from different Unicode versions: against Python 3.11's
unicodedata 14.0 about 1.5k code points differ), so the tables are probed from the library itself, one code point at
a time, and Python's unicodedata is only used to name canonical combining classes:

  norm     cp -> normalized string (removed / space / " cjk " / lowercased and accent-stripped). The normalizer is a
           per-code-point map except for NFD's canonical reordering, which is covered by
  reorder  per code point: 0 = blocks reordering (a starter, or a removed ccc-0 mark), 1..254 = the canonical
           combining class of a kept mark that NFD reorders, 255 = removed and transparent to reordering (removed
           marks with ccc > 0 and characters clean_text drops before NFD);
  punct    BERT punctuation (the pre-tokenizer splits it into its own word);
  space    whitespace (the pre-tokenizer splits on it).

Every probed property is re-checked against the library over all code points before the file is written.
"""
import collections
import pathlib
import sys
import unicodedata as ud

ENC = "sentence-transformers/all-MiniLM-L6-v2"
K_DEL, K_SPACE, K_CJK, K_DELTA, K_DELTA2, K_STR, K_HANGUL = range(7)
KN = ["K_DEL", "K_SPACE", "K_CJK", "K_DELTA", "K_DELTA2", "K_STR", "K_HANGUL"]


def cps():
    return (c for c in range(0x110000) if not 0xD800 <= c < 0xE000)


def hangul_nfd(c):
    s = c - 0xAC00
    L, V, T = 0x1100 + s // 588, 0x1161 + (s % 588) // 28, 0x11A7 + s % 28
    return chr(L) + chr(V) + (chr(T) if T != 0x11A7 else "")


def main():
    from transformers import AutoTokenizer
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parents[1] / "src" / "st_unicode_tab.c")
    bt = AutoTokenizer.from_pretrained(ENC).backend_tokenizer
    N, P = bt.normalizer.normalize_str, bt.pre_tokenizer.pre_tokenize_str
    import tokenizers
    fmap = {}
    for c in cps():
        o = N(chr(c))
        if o != chr(c):
            fmap[c] = o
    f = lambda c: fmap.get(c, chr(c))  # noqa: E731

    # ---- norm map -> ranges
    per = {}
    for c, o in fmap.items():
        if o == "":
            per[c] = (K_DEL, 0)
        elif o == " ":
            per[c] = (K_SPACE, 0)
        elif o == " " + chr(c) + " ":
            per[c] = (K_CJK, 0)
        elif 0xAC00 <= c <= 0xD7A3 and o == hangul_nfd(c):
            per[c] = (K_HANGUL, 0)
        elif len(o) == 1:
            per[c] = (K_DELTA, ord(o) - c)
        else:
            per[c] = (K_STR, o)
    ranges = []  # [lo, hi, kind, arg]
    for c in sorted(per):
        k, a = per[c]
        if k != K_STR and ranges and ranges[-1][1] == c - 1 and ranges[-1][2] == k and ranges[-1][3] == a:
            ranges[-1][1] = c
        else:
            ranges.append([c, c, k, a])
    # alternating case pairs: single DELTA entries at lo, lo+2, ... with identity in between
    merged = []
    for r in ranges:
        p = merged[-1] if merged else None
        if (p and r[2] == K_DELTA and r[0] == r[1] and p[2] in (K_DELTA, K_DELTA2) and p[3] == r[3] and r[0] == p[1] + 2
                and (p[2] == K_DELTA2 or p[0] == p[1]) and (p[1] + 1) not in per):
            p[2], p[1] = K_DELTA2, r[0]
        else:
            merged.append(r)
    ranges = merged
    pool, maxout = [0], 3
    for r in ranges:
        if r[2] == K_STR:
            s = r[3]
            maxout = max(maxout, len(s))
            r[3] = len(pool)
            pool += [len(s)] + [ord(x) for x in s]

    # ---- reordering classes
    kept = [c for c in cps() if f(c) == chr(c) and ud.combining(chr(c))]
    # two reference marks the library really reorders (its NFD tables are older than unicodedata's for some marks):
    # the highest-class and the lowest-class kept mark that swap with each other
    by_cc = sorted(kept, key=lambda c: ud.combining(chr(c)))
    rhi = rlo = None
    for h in reversed(by_cc):
        rlo = next((l for l in by_cc if ud.combining(chr(l)) < ud.combining(chr(h))
                    and N(chr(h) + chr(l)) == chr(l) + chr(h)), None)
        if rlo is not None:
            rhi = h
            break
    lo_c, hi_c = ud.combining(chr(rlo)), ud.combining(chr(rhi))
    reord = {}
    for c in cps():
        o = f(c)
        if o == "":
            if N(chr(rhi) + chr(c) + chr(rlo)) == chr(rlo) + chr(rhi):
                reord[c] = 255
        elif o == chr(c) and (N(chr(rhi) + chr(c)) != chr(rhi) + o or N(chr(c) + chr(rlo)) != o + chr(rlo)):
            reord[c] = None
    refs = {}
    for c in reord:
        if reord[c] is None and ud.combining(chr(c)):
            refs.setdefault(ud.combining(chr(c)), c)
    for c in reord:
        if reord[c] is not None:
            continue
        cc = ud.combining(chr(c))
        if not cc:  # newer than Python's unicodedata: place it between the refs it sorts against
            below = [k for k, r in refs.items() if N(chr(r) + chr(c)) == chr(c) + chr(r)]
            above = [k for k, r in refs.items() if N(chr(c) + chr(r)) == chr(r) + chr(c)]
            eq = [k for k in refs if k not in below and k not in above]
            assert eq, hex(c)
            cc = eq[0]
        reord[c] = cc
    assert 1 <= lo_c and hi_c <= 254
    rr = []
    for c in sorted(reord):
        v = reord[c]
        if rr and rr[-1][1] == c - 1 and rr[-1][2] == v:
            rr[-1][1] = c
        else:
            rr.append([c, c, v])

    def model(s):
        """What st_wordpiece.c does: per-code-point norm, then a stable sort of each run of reorderable output code
        points (class of the output code point itself; removed code points are transparent (255) or end the run)."""
        items = []
        for ch in s:
            o = f(ord(ch))
            if not o:
                if reord.get(ord(ch), 0) != 255:
                    items.append((0, ""))
                continue
            items += [(reord.get(ord(x), 0) % 255, x) for x in o]
        res, run = [], []
        for cl, x in items + [(0, "")]:
            if cl:
                run.append((cl, x))
                continue
            res += [y for _, y in sorted(run, key=lambda t: t[0])] + [x]
            run = []
        return "".join(res)

    # check the model against the library: every reorderable mark against every other, every code point next to
    # the two references (both orders)
    R = [c for c in reord if reord[c] != 255]
    for a in R:
        for b in R:
            assert N(chr(a) + chr(b)) == model(chr(a) + chr(b)), (hex(a), hex(b))
    for c in cps():
        for s in (chr(rhi) + chr(c) + chr(rlo), chr(rlo) + chr(c) + chr(rhi), chr(rhi) + chr(c), chr(c) + chr(rlo)):
            assert N(s) == model(s), [hex(ord(x)) for x in s]

    # ---- pre-tokenizer classes
    punct, space = [], []
    for c in cps():
        r = [w for w, _ in P("a" + chr(c) + "b")]
        if r == ["a", chr(c), "b"]:
            punct.append(c)
        elif r == ["a", "b"]:
            space.append(c)
        else:
            assert r == ["a" + chr(c) + "b"], (hex(c), r)

    def to_ranges(xs):
        o = []
        for c in xs:
            if o and o[-1][1] == c - 1:
                o[-1][1] = c
            else:
                o.append([c, c])
        return o

    pr, sr = to_ranges(punct), to_ranges(space)

    # ---- self-check of the norm table: decode every code point back
    def dec(c):
        import bisect
        i = bisect.bisect_right([r[0] for r in ranges], c) - 1
        if i < 0 or c > ranges[i][1]:
            return chr(c)
        lo, hi, k, a = ranges[i]
        if k == K_DEL:
            return ""
        if k == K_SPACE:
            return " "
        if k == K_CJK:
            return " " + chr(c) + " "
        if k == K_HANGUL:
            return hangul_nfd(c)
        if k == K_DELTA:
            return chr(c + a)
        if k == K_DELTA2:
            return chr(c + a) if (c - lo) % 2 == 0 else chr(c)
        return "".join(chr(x) for x in pool[a + 1:a + 1 + pool[a]])

    for c in cps():
        assert dec(c) == f(c), hex(c)

    # ---- write
    L = []
    w = L.append
    w("/* Generated by tools/gen_unicode.py from tokenizers %s (the HF tokenizer of %s); do not edit.\n"
      " * %d norm ranges, %d pool words, %d reorder ranges, %d punctuation ranges, %d space ranges. */"
      % (tokenizers.__version__, ENC, len(ranges), len(pool), len(rr), len(pr), len(sr)))
    w('#include "st_unicode.h"\n')
    w("enum { K_DEL, K_SPACE, K_CJK, K_DELTA, K_DELTA2, K_STR, K_HANGUL };")
    w("typedef struct {\n  uint32_t lo, hi;\n  int32_t arg;\n  uint8_t kind;\n} norm_range;")
    w("typedef struct {\n  uint32_t lo, hi;\n} cp_range;")
    w("typedef struct {\n  uint32_t lo, hi;\n  uint8_t v;\n} cls_range;\n")

    def emit(name, typ, items, fmt, per_line):
        w("static const %s %s[%d] = {" % (typ, name, len(items)))
        for i in range(0, len(items), per_line):
            w("    " + " ".join(fmt(x) for x in items[i:i + per_line]))
        w("};")

    emit("NORM", "norm_range", ranges, lambda r: "{0x%X,0x%X,%d,%d}," % (r[0], r[1], r[3], r[2]), 4)
    emit("POOL", "uint32_t", pool, lambda x: "0x%X," % x, 12)
    emit("REORDER", "cls_range", rr, lambda r: "{0x%X,0x%X,%d}," % (r[0], r[1], r[2]), 5)
    emit("PUNCT", "cp_range", pr, lambda r: "{0x%X,0x%X}," % (r[0], r[1]), 6)
    emit("SPACE", "cp_range", sr, lambda r: "{0x%X,0x%X}," % (r[0], r[1]), 6)
    w("""
#define FIND(tab, n, c, i)                       \\
  do {                                           \\
    int a_ = 0, b_ = (n)-1;                      \\
    i = -1;                                      \\
    while (a_ <= b_) {                           \\
      int m_ = (a_ + b_) / 2;                    \\
      if ((c) < tab[m_].lo) b_ = m_ - 1;         \\
      else if ((c) > tab[m_].hi) a_ = m_ + 1;    \\
      else { i = m_; break; }                    \\
    }                                            \\
  } while (0)

int st_uni_norm(uint32_t c, uint32_t out[ST_UNI_MAXOUT]) {
  int i;
  if (c < 0x80) {   /* ASCII fast path: controls removed, \\t \\n \\r -> space, A-Z lowercased */
    if (c == 9 || c == 10 || c == 13) { out[0] = ' '; return 1; }
    if (c < 32 || c == 127) return 0;
    out[0] = (c >= 'A' && c <= 'Z') ? c + 32 : c;
    return 1;
  }
  FIND(NORM, (int)(sizeof NORM / sizeof NORM[0]), c, i);
  if (i < 0) { out[0] = c; return 1; }
  const norm_range* r = &NORM[i];
  switch (r->kind) {
    case K_DEL: return 0;
    case K_SPACE: out[0] = ' '; return 1;
    case K_CJK: out[0] = ' '; out[1] = c; out[2] = ' '; return 3;
    case K_DELTA: out[0] = (uint32_t)((int32_t)c + r->arg); return 1;
    case K_DELTA2: out[0] = ((c - r->lo) & 1) ? c : (uint32_t)((int32_t)c + r->arg); return 1;
    case K_HANGUL: {
      uint32_t s = c - 0xAC00, t = s % 28;
      out[0] = 0x1100 + s / 588;
      out[1] = 0x1161 + (s % 588) / 28;
      if (!t) return 2;
      out[2] = 0x11A7 + t;
      return 3;
    }
    default: {
      const uint32_t* p = POOL + r->arg;
      for (uint32_t k = 0; k < p[0]; ++k) out[k] = p[k + 1];
      return (int)p[0];
    }
  }
}

int st_uni_reorder_class(uint32_t c) {
  int i;
  if (c < 0x80) return c < 32 && c != 9 && c != 10 && c != 13 ? 255 : (c == 127 ? 255 : 0);
  FIND(REORDER, (int)(sizeof REORDER / sizeof REORDER[0]), c, i);
  return i < 0 ? 0 : REORDER[i].v;
}

int st_uni_is_punct(uint32_t c) {
  int i;
  FIND(PUNCT, (int)(sizeof PUNCT / sizeof PUNCT[0]), c, i);
  return i >= 0;
}

int st_uni_is_space(uint32_t c) {
  int i;
  FIND(SPACE, (int)(sizeof SPACE / sizeof SPACE[0]), c, i);
  return i >= 0;
}""")
    # the ASCII fast paths must agree with the tables
    for c in range(0x80):
        o = f(c)
        exp = " " if c in (9, 10, 13) else ("" if c < 32 or c == 127 else chr(c).lower())
        assert o == exp, hex(c)
        assert reord.get(c, 0) == (255 if (c < 32 and c not in (9, 10, 13)) or c == 127 else 0), hex(c)
    assert maxout <= 4, maxout
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out}: {len(ranges)} norm ranges, {len(pool)} pool, {len(rr)} reorder, {len(pr)} punct, {len(sr)} space;"
          f" reorder refs ccc {lo_c}..{hi_c}, {len(R)} reorderable marks")


if __name__ == "__main__":
    main()
