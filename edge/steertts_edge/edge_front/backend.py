"""Drop-in replacement for the g2p_en.G2p object the front end uses (cmu, homograph2features, predict, __call__),
built only from the edge assets (no NLTK / g2p_en / num2words / inflect / torch).

Lexicon = "residual" CMUdict: a word whose CMUdict pronunciation the int8 G2P reproduces (with a safe logit margin)
is stored without phones (membership only) or dropped entirely (long words, see build_lexicon.py); everything else
keeps its phones. Lookups therefore return exactly CMUdict's first pronunciation.
"""
import json
import os
import lzma
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

from .g2p_np import G2P
from .hetero import Hetero
from .lexicon import Lexicon
from .tok import word_tokenize

ASSETS = Path(os.environ.get("EDGE_FRONT_ASSETS", str(Path(__file__).resolve().parent / "assets")))


class _Cmu:
    """`w in cmu` / `cmu[w][0]` with g2p_en semantics on top of the residual lexicon. Entries stored without phones
    are words the model reproduces; long model-reproduced words are not stored at all, so for them `w in cmu` is
    False and the caller's fall-through to predict() gives the same phones (only the letter-digit rule's
    `s.lower() in cmu` test on such words can differ; see edge_front_report.md)."""

    def __init__(self, lex, model):
        self.lex, self.model = lex, model

    def __contains__(self, w):
        return isinstance(w, str) and w.isascii() and self.lex.get(w) is not None

    def __getitem__(self, w):
        p = self.lex.get(w) if isinstance(w, str) and w.isascii() else None
        if p is None:
            raise KeyError(w)
        return [p if p else self.model.member(w)]


class _Model:
    def __init__(self, g2p, oov):
        self.g2p, self.oov = g2p, oov
        self.predict = lru_cache(maxsize=16384)(self._predict)
        self.member = lru_cache(maxsize=16384)(lambda w: self.g2p.predict(w))  # phone-less CMUdict entries

    def _predict(self, w):
        p = self.oov.get(w) if w.isascii() else None
        return list(p) if p is not None else self.g2p.predict(w)


class EdgeG2p:
    def __init__(self, assets=ASSETS):
        assets = Path(assets)
        meta = json.loads(lzma.decompress((assets / "front.json.xz").read_bytes()))
        self.meta = meta
        self.homograph2features = {w: (a.split(), b.split(), pos) for w, (a, b, pos) in meta["homographs"].items()}
        if meta.get("postag"):  # pruned NLTK perceptron port (exact algorithm, pruned weights)
            from .postag import PosHetero
            self.hetero = PosHetero(meta["postag"], meta["homographs"])
        else:                   # sparse lexical-context classifier derived from the tagger's decisions
            self.hetero = Hetero(meta.get("hetero"), meta["homographs"])
        if (assets / "gru.npz").exists():   # tier B: quantised copy of g2p_en's own GRU
            from .gru_np import GruG2P
            g2p = GruG2P(assets / "gru.npz")
        else:                               # tier A: tiny int8 transformer (g2p_np.py)
            g2p = G2P(assets / "g2p.npz")
        oovp = assets / "oov_cache.lex"
        oov = Lexicon(oovp.read_bytes() if oovp.exists() else b"")
        self.model = _Model(g2p, oov)
        self.cmu = _Cmu(Lexicon(assets / "cmudict.lex"), self.model)

    def predict(self, w):
        return self.model.predict(w)

    def pron(self, w, toks=None, j=None):
        """g2p_en's per-token rule (non-alpha token -> itself, homograph, CMUdict, neural)."""
        if re.search("[a-z]", w) is None:
            return [w]
        if w in self.homograph2features:
            p1, p2, _ = self.homograph2features[w]
            return p1 if self.hetero.first(toks, j) else p2
        if w in self.cmu:
            return self.cmu[w][0]
        return self.predict(w)

    def __call__(self, text):
        """g2p_en.G2p.__call__ (g2p_en front-end path)."""
        from .normnum import normalize_numbers
        text = normalize_numbers(text)
        text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
        text = text.lower()
        text = re.sub("[^ a-z'.,?!\\-]", "", text)
        text = text.replace("i.e.", "that is").replace("e.g.", "for example")
        words = word_tokenize(text)
        prons = []
        for j, w in enumerate(words):
            prons.extend(self.pron(w, words, j))
            prons.extend([" "])
        return prons[:-1]
