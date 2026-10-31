"""Agreement of the v1 Python front end (steertts.text: g2p_en + NLTK) with the edge front end (steertts_edge.text,
== the C front end, == golden.jsonl ids) on the golden to_phonemes texts. Documentation only.

  PYTHONPATH=<repo>:<repo>/edge CUDA_VISIBLE_DEVICES= python front_v1_vs_edge.py [../../native/tests/front/golden.jsonl]
"""
import collections
import difflib
import json
import re
import sys

from steertts import text as v1

path = sys.argv[1] if len(sys.argv) > 1 else "../../native/tests/front/golden.jsonl"
rows = [j for j in map(json.loads, open(path, encoding="utf-8")) if j["k"] == "ph" and "error" not in j]
SYM = v1.SYMBOLS


def words(ids, widx):
    out = collections.OrderedDict()
    for i, w in zip(ids, widx):
        if w >= 0:
            out.setdefault(w, []).append(SYM[i])
    return [tuple(v) for v in out.values()]


same_ids = same_widx = same_nwords = 0
n_words = matched = 0
diffs = collections.Counter()
examples = {}
for j in rows:
    ids, widx, _ = v1.to_phonemes(j["text"])
    ids, widx = list(map(int, ids)), list(map(int, widx))
    same_ids += ids == j["ids"]
    same_widx += ids == j["ids"] and widx == j["widx"]
    a, b = words(j["ids"], j["widx"]), words(ids, widx)
    same_nwords += len(a) == len(b)
    n_words += len(a)
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    matched += sum(m.size for m in sm.get_matching_blocks())
    toks = re.findall(r"[A-Za-z][A-Za-z']*|\d+", j["norm"])
    for op, i1, i2, k1, k2 in sm.get_opcodes():
        if op == "replace" and i2 - i1 == k2 - k1:
            for d in range(i2 - i1):
                w = toks[i1 + d].lower() if len(toks) == len(a) else "?"
                key = (w, " ".join(a[i1 + d]), " ".join(b[k1 + d]))
                diffs[key] += 1
                examples.setdefault(key, j["text"])

n = len(rows)
print(f"texts: {n}")
print(f"identical ids:            {same_ids} ({100 * same_ids / n:.1f}%)")
print(f"identical ids + word_idx: {same_widx} ({100 * same_widx / n:.1f}%)")
print(f"same word count:          {same_nwords} ({100 * same_nwords / n:.1f}%)")
print(f"per-word agreement:       {matched}/{n_words} edge words ({100 * matched / n_words:.2f}%) (difflib alignment)")
print("\nmost frequent word differences (word: edge | v1  x count):")
for (w, e, v), c in diffs.most_common(25):
    print(f"  {w:>14}: {e:<28} | {v:<28} x{c}")
