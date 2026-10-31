"""English number words without num2words / inflect (edge front end).

`cardinal(n)` / `ordinal(n)` reproduce num2words(n) / num2words(n, to="ordinal") (lang en) exactly, including its
hyphens ("forty-two"), the British "and" and the ", " between thousands groups, so callers keep their own clean-up.
`cardinal_float(x)` reproduces num2words(<float>) (g2p_en front-end path). `inflect_words(n, ...)` reproduces the
inflect.engine().number_to_words calls g2p_en's normalize_numbers makes (g2p_en front-end path only).
Property-tested in tests/test_edge_front.py against the real libraries.
"""
from decimal import Decimal
import math

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
         "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
# short-scale names num2words EN knows (10**6 ... 10**(3*len+3)); MAXVAL = 1000 * the largest
_HIGH = ["m", "b", "tr", "quadr", "quint", "sext", "sept", "oct", "non", "dec",
         "undec", "duodec", "tredec", "quattuordec", "quindec", "sexdec", "septendec", "octodec", "novemdec", "vigint",
         "unvigint", "duovigint", "trevigint", "quattuorvigint", "quinvigint", "sexvigint", "septenvigint",
         "octovigint", "novemvigint", "trigint", "untrigint", "duotrigint"]
_SCALES = ["", "thousand"] + [h + "illion" for h in _HIGH]
MAXVAL = 1000 * 10 ** (3 * (len(_SCALES) - 1))


def _below100(n):
    if n < 20:
        return _ONES[n]
    t, o = divmod(n, 10)
    return _TENS[t] + ("-" + _ONES[o] if o else "")


def _below1000(n, andword=" and "):
    h, r = divmod(n, 100)
    if not h:
        return _below100(r)
    return _ONES[h] + " hundred" + ((andword + _below100(r)) if r else "")


def cardinal(n):
    """num2words(int n) (lang en)."""
    n = int(n)
    if n < 0:
        return "minus " + cardinal(-n)
    if n >= MAXVAL:
        raise OverflowError(f"abs({n}) must be less than {MAXVAL}.")
    if n == 0:
        return "zero"
    groups = []
    k = 0
    while n:
        n, g = divmod(n, 1000)
        groups.append((g, k))
        k += 1
    parts = []
    for g, k in reversed(groups):
        if not g:
            continue
        parts.append((_below1000(g), k))
    # num2words joins groups with ", "; a final group below 100 after a higher group gets " and "
    out = ""
    for j, (w, k) in enumerate(parts):
        s = w + (" " + _SCALES[k] if k else "")
        if j == 0:
            out = s
        elif k == 0 and groups[0][0] < 100:
            out += " and " + s
        else:
            out += ", " + s
    return out


_ORD = {"one": "first", "two": "second", "three": "third", "four": "fourth", "five": "fifth", "six": "sixth",
        "seven": "seventh", "eight": "eighth", "nine": "ninth", "ten": "tenth", "eleven": "eleventh",
        "twelve": "twelfth"}


def _ord_word(w):
    if w in _ORD:
        return _ORD[w]
    if w.endswith("y"):
        w = w[:-1] + "ie"
    return w + "th"


def ordinal(n):
    """num2words(int n, to="ordinal") (lang en)."""
    n = int(n)
    if n < 0:
        raise TypeError("Cannot treat negative num %s as ordinal." % n)
    ws = cardinal(n).split(" ")
    last = ws[-1].split("-")
    last[-1] = _ord_word(last[-1].lower())
    ws[-1] = "-".join(last)
    return " ".join(ws)


def cardinal_float(value):
    """num2words(<float>) (lang en): integer part, 'point', then each decimal digit (precision from str(value))."""
    value = float(value)
    if value == int(value) and math.isfinite(value):
        return cardinal(int(value))
    pre = int(value)
    precision = abs(Decimal(str(value)).as_tuple().exponent)
    post = abs(value - pre) * 10 ** precision
    post = int(round(post)) if abs(round(post) - post) < 0.01 else int(math.floor(post))
    post = str(post)
    post = "0" * (precision - len(post)) + post
    out = [cardinal(pre)]
    if precision:
        out.append("point")
    for i in range(precision):
        out.append(cardinal(int(post[i])))
    return " ".join(out)


# ---- inflect.number_to_words subset (g2p_en.expand.normalize_numbers; g2p_en path only) -------------------------
_IONES = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
_ITEENS = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_IMILL = [" ", " thousand", " million", " billion", " trillion", " quadrillion", " quintillion", " sextillion",
          " septillion", " octillion", " nonillion", " decillion"]


def _i_tenfn(tens, units, mindex=0):
    if tens != 1:
        tw = _TENS[tens] if tens else ""
        if tens and units:
            hy = "-"
        else:
            hy = ""
        return f"{tw}{hy}{_IONES[units]}{_IMILL[mindex] if (tens or units) else ''}"
    return f"{_ITEENS[units]}{_IMILL[mindex]}"


def _i_hundfn(h, t, u, mindex, andword):
    if h:
        return f"{_IONES[h]} hundred{andword if t or u else ''}{_i_tenfn(t, u)}{_IMILL[mindex]}, "
    if t or u:
        return f"{_i_tenfn(t, u)}{_IMILL[mindex]}, "
    return ""


def _i_enword(num, group, zero, one, andword):
    import re
    if group == 1:
        num = re.sub(r"(\d)", lambda m: (_IONES[int(m.group(1))] or zero) + ", ", num)
    elif group == 2:
        num = re.sub(r"(\d)(\d)", lambda m: (_i_tenfn(int(m.group(1)), int(m.group(2))) if int(m.group(1)) else
                                             (f"{zero} {_IONES[int(m.group(2))] or zero}" if True else "")) + ", ", num)
        num = re.sub(r"(\d)$", lambda m: (_IONES[int(m.group(1))] or zero) + ", ", num)
    elif group == 3:
        raise NotImplementedError
    elif int(num) == 0:
        num = zero
    elif int(num) == 1:
        num = one
    else:
        num = num.lstrip().lstrip("0")
        mill = 0
        # hundreds triples from the right
        out = []
        while len(num) > 3:
            num, trip = num[:-3], num[-3:]
            h, t, u = int(trip[0]), int(trip[1]), int(trip[2])
            out.insert(0, _i_hundfn(h, t, u, mill, andword))
            mill += 1
        trip = num.rjust(3, "0")
        out.insert(0, _i_hundfn(int(trip[0]), int(trip[1]), int(trip[2]), mill, andword))
        num = "".join(out)
    return num


def inflect_words(n, andword="and", zero="zero", group=0):
    """inflect.engine().number_to_words(n, andword=..., zero=..., group=...) for non-negative ints (str or int).
    Accepts ordinal strings like '21st' (as normalize_numbers passes them)."""
    import re
    s = str(n)
    ordsuf = ""
    m = re.match(r"^(\d+)(st|nd|rd|th)$", s)
    if m:
        s, ordsuf = m.group(1), m.group(2)
    aw = f" {andword} " if andword else " "
    if group:
        chunks = _i_enword(s, group, zero, "one", aw)
    else:
        chunks = _i_enword(s, 0, zero, "one", aw)
    chunks = re.sub(r"(\S)\s+,", r"\1,", chunks)
    chunks = re.sub(r"\s+,", ",", chunks)
    chunks = re.sub(r",\s*$", "", chunks)
    chunks = re.sub(r"\s+", " ", chunks).strip()
    if group:
        out = ", ".join(c.strip() for c in chunks.split(",") if c.strip())
    else:
        # inflect: a final single-word group after a comma is joined with the and-word ("one thousand and one")
        out = re.sub(r"\s+", " ", re.sub(r", (\S+)\Z", f" {andword} \\1", chunks)).strip()
    if ordsuf:
        out = _inflect_ordinal_words(out)
    return out


def _inflect_ordinal_words(words):
    import re
    m = re.search(r"(\w+)$", words)
    if not m:
        return words
    w = m.group(1)
    o = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth",
         "twelve": "twelfth"}
    if w in o:
        nw = o[w]
    elif w.endswith("ty"):
        nw = w[:-1] + "ieth"
    else:
        nw = w + "th"
    return words[:m.start()] + nw
