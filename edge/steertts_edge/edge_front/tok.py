"""NLTK TweetTokenizer().tokenize restricted to g2p_en's cleaned alphabet [ a-z'.,?!-] (pure `re`).

g2p_en lower-cases and strips everything else before tokenising, so only these TweetTokenizer branches can fire:
naked-domain URLs ("mr.smith" / "end.next" stay one token), words with inner apostrophes / hyphens, plain words,
ellipses (". . ." included) and single other characters; runs of >= 4 equal non-letters are first cut to 3
(HANG_RE). Fuzz-tested against nltk in tests/test_edge_front.py.
"""
import re

_HANG = re.compile(r"([^a-z0-9])\1{3,}")
_WORD = re.compile(
    r"(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+){0,126}[.](?:[a-z]{2,13})\b/?(?!@)"  # naked domain
    r"|[a-z](?:[a-z]|['\-_])+[a-z]"                                       # words with apostrophes or dashes
    r"|[a-z0-9_]+"                                                        # plain words
    r"|\.(?:\s*\.)+"                                                      # ellipsis
    r"|\S"
)


def word_tokenize(text):
    return _WORD.findall(_HANG.sub(r"\1\1\1", text))
