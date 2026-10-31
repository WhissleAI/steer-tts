"""Golden data for tests/test_prompt.c (the MiniLM tokenizer, sentence embedding and prompt mapper).

    python tests/make_golden_prompt.py [out_dir]            # default tests/golden/prompt
    python tests/make_golden_prompt.py --tok FILE N SEED    # only a tokenizer corpus of N texts (large-scale check)

Files (UTF-8 text is stored hex-encoded so any byte sequence survives):
    tokens.txt          hex(text) \\t ids (comma separated)           tokenizer corpus (HF tokenizer, max_length 256)
    prompts.txt         hex(prompt), one per line                      the mapping prompts
    emb_fp32.f32        [n_prompts, 384] float32                       MiniLM embeddings (fp32 encoder)
    emb_int8.f32        [n_prompts, 384] float32                       MiniLM embeddings (round_int8_ encoder)
    map_fp32.txt        per prompt: top tags + rounded p, ctrl dims + rounded z, then the unrounded values
    map_int8.txt        the same for SteerTTS(precision="int8").prompt_mapper
Needs the steertts package, transformers and the cached models (HF_HUB_OFFLINE=1); weights dir from STEERTTS_V1_MODEL.
"""
import os
import pathlib
import random
import sys

import numpy as np
import torch

V1 = os.environ.get("STEERTTS_V1_MODEL", "../models/steer-tts")
ENC = "sentence-transformers/all-MiniLM-L6-v2"


def hexs(s):
    return s.encode("utf-8").hex()


# ------------------------------------------------------------------ tokenizer corpus
POOLS = {
    "ascii": [chr(c) for c in range(32, 127)],
    "ctrl": [chr(c) for c in list(range(0, 32)) + [127, 0x85, 0xAD, 0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF, 0xFFFD, 0x202E,
                                                     0x2028, 0x2029, 0xE000, 0xF8FF, 0x1D173]],
    "space": [" ", "\t", "\n", "\r", " ", " ", " ", " ", " ", " ", " ", " ",
              "　", "\u000b", "\u000c", "\u001c", "\u0085"],
    "latin": [chr(c) for c in range(0xA0, 0x250)] + [chr(c) for c in range(0x1E00, 0x1F00)],
    "comb": [chr(c) for c in list(range(0x300, 0x370)) + list(range(0x1AB0, 0x1AD0)) + list(range(0x1DC0, 0x1E00))
             + list(range(0x20D0, 0x20F1)) + [0x8D4, 0x8D5, 0x1B44, 0x1BAA, 0x302E, 0x1D165, 0x1D166, 0x1D16D, 0x093C,
                                              0x0F71, 0x0F72, 0x0F74, 0x0F80, 0x05B0, 0x05B4, 0x064B, 0x0651, 0x0E38]],
    "greek": [chr(c) for c in range(0x370, 0x400)] + [chr(c) for c in range(0x1F00, 0x2000)],
    "cyr": [chr(c) for c in range(0x400, 0x530)],
    "semitic": [chr(c) for c in range(0x591, 0x700)],
    "indic": [chr(c) for c in range(0x900, 0xE80)],
    "cjk": [chr(c) for c in list(range(0x4E00, 0x4F00)) + list(range(0x3040, 0x3100)) + list(range(0xF900, 0xFB00))
            + list(range(0x20000, 0x20040)) + list(range(0x2F800, 0x2F820)) + list(range(0x3400, 0x3440))],
    "hangul": [chr(c) for c in list(range(0xAC00, 0xAD00)) + list(range(0x1100, 0x1200)) + [0xD7A3, 0x3131, 0x314F]],
    "sym": [chr(c) for c in list(range(0x2010, 0x2070)) + list(range(0x2100, 0x2200)) + list(range(0x2200, 0x2300))
            + list(range(0x3000, 0x3040)) + list(range(0xFF00, 0xFFF0)) + list(range(0xFE30, 0xFE70)) + [0xA1, 0xBF, 0xA7, 0xB6]],
    "emoji": [chr(c) for c in list(range(0x1F300, 0x1F650)) + list(range(0x1F900, 0x1FA00)) + list(range(0x1F1E6, 0x1F200))
              + [0x2764, 0xFE0F, 0x200D, 0x1F3FB, 0x1F3FF, 0x2600, 0x2B50]],
    "quotes": ["‘", "’", "“", "”", "«", "»", "‹", "›", "„", "…", "–",
               "—", "'", '"', "`", "´"],
    "special": ["[MASK]", "[CLS]", "[SEP]", "[PAD]", "[UNK]", "[mask]", "[MAS", "[[SEP]]", "[ CLS]"],
    "case": [chr(c) for c in [0x130, 0x131, 0x1E9E, 0xDF, 0x3A3, 0x3C2, 0x1C4, 0x1C5, 0x1C6, 0x1F0, 0x149, 0x2126, 0x212A,
                              0x212B, 0xFB00, 0xFB01, 0xFB03, 0x1E96, 0x587, 0x10400, 0x10428, 0x1E900, 0x2C2F, 0xA7C0,
                              0x1C89, 0x1C8A, 0xA7CB, 0xA7DC, 0x10D50]],
}
WORDS = ("speak softly with a warm smile excited faster slower whisper angry sad happy tired bored calm loud deep high "
         "pitch tone voice narrator story café naïve résumé façade jalapeño Zoë Ångström São Paulo Kraków Москва "
         "Ελλάδα 東京 서울 مرحبا שלום नमस्ते ไทย unbelievably antidisestablishmentarianism").split()


def rand_text(rng):
    r = rng.random()
    if r < 0.15:   # word salad with punctuation
        n = rng.randint(1, 30)
        return " ".join(rng.choice(WORDS) + rng.choice(["", "", ",", ".", "!", "?", "...", "'s", "-"]) for _ in range(n))
    if r < 0.25:   # fully random code points
        return "".join(chr(rng.choice([rng.randrange(0x110000), rng.randrange(0x3000), rng.randrange(0x20000)]))
                       for _ in range(rng.randint(1, 40))).encode("utf-8", "replace").decode("utf-8", "replace")
    if r < 0.30:   # long word (> 100 chars) or long word made of valid pieces
        w = "".join(rng.choice("abcdefghij") for _ in range(rng.randint(90, 130)))
        return rng.choice(["", "the ", "Ok, "]) + w + rng.choice(["", " end", "!"])
    if r < 0.35:   # long text beyond 256 tokens
        return " ".join(rng.choice(WORDS) for _ in range(rng.randint(150, 400)))
    if r < 0.40:   # base letter + many combining marks
        return " ".join(rng.choice("aeoAEOncαа") + "".join(rng.choice(POOLS["comb"]) for _ in range(rng.randint(1, 6)))
                        for _ in range(rng.randint(1, 8)))
    # mixture of pools
    pools = rng.sample(sorted(POOLS), rng.randint(1, 4))
    out = []
    for _ in range(rng.randint(1, 50)):
        p = POOLS[rng.choice(pools + ["ascii"])]
        out.append(rng.choice(p))
        if rng.random() < 0.2:
            out.append(" ")
    return "".join(out)


FIXED = ["", " ", "\t\n", "hello", "Hello, World!", "[CLS]", "[MASK] is [SEP] here", "a[MASK]b", "don't stop",
         "é", "é", "İstanbul", "ΟΔΟΣ", "\U0001F600\U0001F44D\U0001F3FD",
         "\U0001F468‍\U0001F469‍\U0001F467", "\U0001F1FA\U0001F1F8", "“quoted” ‘single’",
         "你好世界", "Ｈｅｌｌｏ", "ﬁne", "x" * 100, "x" * 101, "\U0001D165\u0301\U0001D166", "é" * 101,
         "a\u0000b\u0001c", "�", "ᴖ5", "\U0001D15E\U0001D165", "क़ु", "اَلّ",
         "กุ่", "ㄱㅏ 가힣", "¿Qué tal?", "A.B.C.", "50% off - $3.99 #1 @me",
         "slow...  and   SOFT!!!", "Αθήνα", "Привет, мир"]


def tok_corpus(n, seed):
    rng = random.Random(seed)
    return FIXED + [rand_text(rng) for _ in range(max(0, n - len(FIXED)))]


def write_tok(path, texts, tok):
    with open(path, "w") as f:
        for t in texts:
            ids = tok(t, truncation=True, max_length=256)["input_ids"]
            f.write(hexs(t) + "\t" + ",".join(map(str, ids)) + "\n")


# ------------------------------------------------------------------ prompts
HAND = [
    "excited, a bit faster", "soft and sad", "angry and loud", "whisper it slowly", "calm, warm narrator voice",
    "speak like you just won the lottery", "tired and bored, monotone", "high pitched and cheerful",
    "deep low voice, very slow", "nervous, hesitant, a little shaky", "sarcastic", "a mysterious storyteller",
    "happy", "sad", "fast", "slow", "loud", "quiet", "whispering", "laughing while talking", "in pain",
    "panicked and breathless", "proud and confident", "apologetic and soft", "disgusted", "surprised!",
    "full of wonder and amazement", "contemptuous sneer", "nostalgic and wistful", "relieved after a long day",
    "sympathetic, gentle, caring", "skeptical, unconvinced", "frustrated and irritated", "determined",
    "enunciate every word clearly", "emphasize the important words", "cute and playful", "seductive and slow",
    "serene, peaceful, meditative", "embarrassed and quiet", "guilty", "ecstatic!!!", "very very fast and high",
    "slightly slower than normal", "a touch louder please", "lower pitch, same speed", "neutral reading",
    "news anchor, professional", "bedtime story for kids", "sports commentator, energetic", "robotic, flat",
    "", "   ", "SHOUTING IN ALL CAPS", "Speak Softly", "speak    softly", "café-style chill vibe",
    "naïve and innocent", "😀 happy", "sad 😢", "“excited” – faster", "triste y lento", "très heureux",
    "fröhlich und schnell", "грустно и медленно", "悲しい", "开心", "a" * 120,
    "slow, " * 60, "the quick brown fox jumps over the lazy dog " * 30,
    "excited, a bit faster.", "Excited, a bit faster", "excited , a bit faster", "a bit faster, excited",
    "[MASK] voice", "faster faster faster", "not angry", "not happy at all", "less excited", "more excited",
]
ADJ = ["happy", "sad", "angry", "excited", "calm", "tired", "nervous", "confident", "bored", "scared", "joyful", "gloomy",
       "cheerful", "furious", "anxious", "relaxed", "sleepy", "playful", "serious", "dramatic", "gentle", "harsh",
       "warm", "cold", "breathy", "raspy", "bright", "dark", "shy", "bold"]
MOD = ["", "very ", "a bit ", "slightly ", "extremely ", "somewhat ", "really "]
CTRL = ["", "faster", "slower", "louder", "softer", "higher pitch", "lower pitch", "more expressive", "monotone",
        "a bit faster", "much slower", "quietly", "with a deep voice", "with a high voice", "quickly", "at normal speed"]


def prompts(n, seed=1):
    rng = random.Random(seed)
    out = list(HAND)
    while len(out) < n:
        a = rng.choice(MOD) + rng.choice(ADJ)
        r = rng.random()
        if r < 0.4:
            p = a + (", " + rng.choice(CTRL) if rng.random() < 0.7 else "")
        elif r < 0.7:
            p = rng.choice(["speak ", "say it ", "read this ", "talk ", ""]) + rng.choice(CTRL + [a]) + rng.choice(["", " and " + a, ", " + a])
        else:
            p = "a " + a + " " + rng.choice(["voice", "narrator", "tone", "person", "child", "old man", "woman"]) + rng.choice(["", ", " + rng.choice(CTRL)])
        p = p.strip(" ,")
        if p and p not in out:
            out.append(p)
    return out


def map_lines(pm, ps, dims):
    lines = []
    for p in ps:
        r = pm(p)
        key = " ".join(p.lower().split())
        tp, sp, v = pm.cache[key]
        top = torch.topk(tp, 3)
        name2i = {t: i for i, t in enumerate(pm.tags)}
        f1 = ";".join(f"{name2i[t]}:{round(pv * 1000)}" for t, pv in r["top_tags"])
        f2 = ";".join(f"{dims.index(d)}:{round(z * 100)}" for d, z in r["ctrl"].items())
        raw = (" ".join(repr(float(x)) for x in top.values) + " | " + " ".join(repr(float(x)) for x in sp) + " | "
               + " ".join(repr(float(x)) for x in v))
        lines.append(f"{f1}\t{f2}\t{raw}")
    return lines


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--tok":
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(ENC)
        write_tok(sys.argv[2], tok_corpus(int(sys.argv[3]), int(sys.argv[4])), tok)
        return
    torch.set_num_threads(4)
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parent / "golden" / "prompt")
    out.mkdir(parents=True, exist_ok=True)
    from steertts import SteerTTS
    from steertts.config import CONTROL_DIMS
    ps = prompts(200)
    with open(out / "prompts.txt", "w") as f:
        for p in ps:
            f.write(hexs(p) + "\n")
    for prec in ("fp32", "int8"):
        tts = SteerTTS(V1, device="cpu", flow=False, precision=prec)
        pm = tts.prompt_mapper
        if prec == "fp32":
            write_tok(out / "tokens.txt", tok_corpus(1500, 12345) + ps, pm.enc.tok)
        emb = torch.cat([pm.enc([p]) for p in ps]).numpy().astype(np.float32)
        emb.tofile(out / f"emb_{prec}.f32")
        (out / f"map_{prec}.txt").write_text("\n".join(map_lines(pm, ps, CONTROL_DIMS)) + "\n")
        print(prec, "done", emb.shape)


if __name__ == "__main__":
    main()
