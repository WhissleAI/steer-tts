"""Heteronym disambiguation without NLTK: a sparse lexical-context classifier derived from the NLTK
averaged-perceptron tagger's decisions on g2p_en's homograph list.

g2p_en reads homograph w as pron1 iff pos_tag(tokens)[j].startswith(pos1). The edge model scores that decision
directly from the surrounding tokens: score = sum of int8 weights of the active features, pron1 iff score > 0.
Features are keyed both by the word ("w:read|...") and by its pos1 class ("g:VBD|...") so homographs never seen in
the fitting text still get the class-level context rules.
"""

_SUF = 2


def _suf(t):
    return t[-_SUF:] if t.isalpha() and len(t) > _SUF else t


def features(toks, j, tagdict=None):
    """Context features of toks[j] (toks = the lower-cased token list g2p_en would tag). tagdict: {word: coarse
    tag} (most frequent NLTK tag, 2 letters) for frequent words; others get a suffix class."""
    def at(k):
        if k < 0:
            return "<s>"
        if k >= len(toks):
            return "</s>"
        return toks[k]
    p1, p2, n1, n2 = at(j - 1), at(j - 2), at(j + 1), at(j + 2)
    return ["b", "p1=" + p1, "n1=" + n1, "p2=" + p2, "n2=" + n2, "p1s=" + _suf(p1), "n1s=" + _suf(n1),
            "p1n1=" + p1 + "_" + n1, "p2p1=" + p2 + "_" + p1, "n1n2=" + n1 + "_" + n2, "p2s=" + _suf(p2),
            "n2s=" + _suf(n2)] + (_tagf(p2, p1, n1, n2, tagdict) if tagdict is not None else [])


def _tag(t, td):
    if t in td:
        return td[t]
    if t in ("<s>", "</s>"):
        return t
    if not any(c.isalpha() for c in t):
        return "P"
    return "U" + t[-2:]


def _tagf(p2, p1, n1, n2, td):
    a, b, c, d = _tag(p2, td), _tag(p1, td), _tag(n1, td), _tag(n2, td)
    return ["p1t=" + b, "n1t=" + c, "p2t=" + a, "n2t=" + d, "p1tn1t=" + b + "_" + c, "p2tp1t=" + a + "_" + b,
            "n1tn2t=" + c + "_" + d, "p1n1t=" + p1 + "_" + c, "p1tn1=" + b + "_" + n1]


class Hetero:
    def __init__(self, model, homographs):
        """model: {"scale": float, "w": {feature_key: int}}; homographs: {word: [pron1, pron2, pos1]}."""
        self.w = model.get("w", {}) if model else {}
        self.td = model.get("tags") if model else None
        self.hom = homographs

    def first(self, toks, j):
        """True -> pron1 (the reading g2p_en gives when the POS starts with pos1)."""
        w = toks[j]
        pos1 = self.hom[w][2]
        s = 0
        get = self.w.get
        for f in features(toks, j, self.td):
            s += get("w:" + w + "|" + f, 0) + get("g:" + pos1 + "|" + f, 0)
        return s > 0
