#!/usr/bin/env python3
"""Generate the C++ front-end golden corpus from the Python edge front end (the source of truth).

usage: make_front_golden.py <pkg_dir> <out_dir> [--g2p-words N]

Writes <out_dir>/golden.jsonl, one JSON object per line:
  {"k": "ph", "text": s, "norm": normalize(s) | null, "ids": [...], "widx": [...]}      text.to_phonemes(s)[:2]
      (+ "error": "<ExcType>: <message>" when to_phonemes raises; "norm_error" when normalize raises)
  {"k": "mk", "text": s, "tag": t|null, "ctrl": [[k, v], ...]|null, "segs": [seg, ...]}  MarkupSynth.parse
      seg = {"text", "tag", "ctrl": [[k, v]...] (dict order), "ws": [[word, pitch, energy, dur], ...] (missing keys
      filled with 0/0/1), "pause_ms", "event"}; a style tuple in pause_ms / event (Python quirk) is written as
      -1 / "<style-tuple>". (+ "error" when parse raises)
and <out_dir>/golden_g2p.jsonl: {"w": word, "p": [GRU phoneme indices], "gap": min top1-top2 logit gap} for the
residual lexicon's member-only words (resolved by the GRU at runtime) and random pseudo-words.

Text sources (read-only): seed-tts-eval en/zh meta lists and the CV3 leaderboard texts, plus generated cases
(numbers, money, units, acronyms, heteronyms, unicode, events, fuzz). Deterministic (fixed seeds).
"""
import json
import lzma
import os
import random
import sys
from pathlib import Path

# Optional real-text sources; missing ones are skipped. SEEDTTS_TESTSET = the seed-tts-eval seedtts_testset dir,
# CV3_TEXTS = a JSONL file with a "text" field per line (CV3-Eval zero-shot en).
SEEDTTS = Path(os.environ.get("SEEDTTS_TESTSET", "seedtts_testset"))
SEED_EN = SEEDTTS / "en"
SEED_ZH = SEEDTTS / "zh"
CV3 = Path(os.environ.get("CV3_TEXTS", "cv3_eval_zero_shot_en.jsonl"))


def real_texts():
    en, zh = [], []
    for f in ("meta.lst", "non_para_reconstruct_meta.lst"):
        p = SEED_EN / f
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                parts = line.split("|")
                if len(parts) >= 4:
                    en += [parts[1], parts[3]]
    if CV3.exists():
        for line in CV3.read_text(encoding="utf-8").splitlines():
            en.append(json.loads(line)["text"])
    p = SEED_ZH / "meta.lst"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines()[:300]:
            parts = line.split("|")
            if len(parts) >= 4:
                zh.append(parts[3])
    seen, out = set(), []
    for t in en:
        if t not in seen:
            seen.add(t); out.append(t)
    return out, zh


NAMES = ["Zorbleth", "Kowalczyk", "Nguyen", "Siobhan", "Aoife", "Xochitl", "Saoirse", "Tchaikovsky", "Dvorak",
         "Przybylski", "Oyelaran", "Bjorkman", "Thibodeaux", "Quvenzhane", "Leicester", "Worcestershire", "Goethe",
         "Nietzsche", "Mbappe", "Szczepanski", "Ouagadougou", "Reykjavik", "Llanfair", "Cholmondeley", "Featherstonehaugh",
         "Zhang", "Xiaoming", "Ramaswamy", "Vijayaraghavan", "Okonkwo", "Adebayo", "Kyrgyzstan", "Tbilisi",
         "Chiaroscuro", "Hrothgar", "Eyjafjallajokull", "Quetzalcoatl", "Brzezinski", "Wojciechowski", "Aelfric"]
WORDS = ("the of and to in is was for on that with as by at from it his an were are which this be or has had not "
         "but one their also its first new after have who they two her she been other when there all during into "
         "school time may years more most only over city some world would where later up such used many can state "
         "about national out known university united then made people house record lead read live wind tear bass "
         "close desert minute object present project refuse row sow wound bow produce content conduct permit "
         "contract invalid moderate separate estimate graduate associate").split()


def rand_word(r):
    cons, vow = "bcdfghjklmnpqrstvwxyz", "aeiou"
    n = r.randint(2, 6)
    w = "".join(r.choice(cons) + r.choice(vow) + (r.choice(cons) if r.random() < 0.4 else "") for _ in range(n // 2 + 1))
    if r.random() < 0.15:
        w += r.choice(["ification", "ness", "ology", "ism", "ville", "berg", "stein", "ski", "ton"])
    return w


def gen_number_cases(r):
    out = []
    for _ in range(250):
        k = r.randint(1, 22)
        n = r.randint(10 ** (k - 1), 10 ** k - 1) if k > 1 else r.randint(0, 9)
        out.append(r.choice(["{} people came.", "It costs {}.", "Call {} now", "{}", "Page {} of 900", "about {} km"])
                   .format(f"{n:,}" if r.random() < 0.4 else str(n)))
    for _ in range(120):
        d = f"{r.randint(0, 99999)}.{r.randint(0, 999999):0{r.randint(1, 6)}d}"
        out.append(r.choice(["The value is {}.", "{} percent", "pi is {}", "{}%", "-{} degrees", "v{}"]).format(d))
    for _ in range(80):
        n = r.randint(1, 20000)
        out.append(r.choice(["the {}th time", "on the {}st", "her {}nd try", "the {}rd day", "{}th", "({}th)"]).format(n))
    for y in r.sample(range(1000, 3000), 120):
        out.append(r.choice(["In {} we moved.", "the year {}", "{}-{} season", "since {},", "{}s music"]).format(y, y + r.randint(1, 9)))
    for _ in range(60):
        h, m = r.randint(0, 23), r.randint(0, 59)
        out.append(r.choice(["at {}:{:02d}", "{}:{:02d} pm", "{}:{:02d}:{:02d}", "from {}:{:02d} to 5:00"]).format(h, m, r.randint(0, 59)))
    for _ in range(60):
        out.append(r.choice(["{}/{}/{}", "{}-{:02d}-{:02d}", "on {}.{}.{}", "Jan {}, {}"]).format(
            r.randint(1, 31), r.randint(1, 12), r.randint(1900, 2030)))
    cur = ["$", "£", "€"]
    for _ in range(160):
        c = r.choice(cur)
        a = r.choice([str(r.randint(0, 999)), f"{r.randint(0, 9999999):,}", f"{r.randint(0, 999)}.{r.randint(0, 99):02d}",
                      f"{r.randint(1, 999)}.{r.randint(0, 9)}", f".{r.randint(0, 99):02d}", "0.01", "1", "1.01", "1,000,000"])
        sfx = r.choice(["", " million", " billion", " thousand", "million", " thousands", ",", ".", " each", "/hr"])
        out.append(r.choice(["It cost {}{}{}.", "{}{}{}", "about {} {}{}", "pay {}{}{} now"]).format(c, a, sfx))
    for _ in range(60):
        out.append(r.choice(["{}%", "{} %", "{}.5%", "{},000%", "up {}% from {}%"]).format(r.randint(0, 200), r.randint(0, 99)))
    for _ in range(80):
        a, b = r.randint(0, 2000), r.randint(0, 2000)
        out.append(r.choice(["{}-{}", "{} - {}", "{}–{}", "pages {}-{}", "{}x{}", "{} x {}", "{}×{}", "{}-{}-{}"]).format(a, b, r.randint(0, 99)))
    for _ in range(80):
        out.append(r.choice(["call 555-{:04d}", "({:03d}) {:03d}-{:04d}", "+1 {:03d} {:03d} {:04d}", "{:03d}.{:03d}.{:04d}",
                             "my PIN is {:04d}", "zip {:05d}", "card 4111 1111 1111 {:04d}"]).format(
            r.randint(0, 9999), r.randint(0, 999), r.randint(0, 9999)))
    units = ["km/h", "km/hr", "m/s", "mph", "km", "kg", "cm", "mm", "ft", "lbs", "lb", "kHz", "Hz", "GB", "MB", "km h", "KM", "kgs"]
    for _ in range(140):
        u = r.choice(units)
        out.append(r.choice(["{} {}", "{}{}", "going {} {}.", "{} {} away", "{}{}/s"]).format(r.choice([r.randint(0, 999), f"{r.randint(0, 99)}.{r.randint(0, 9)}"]), u))
    out += ["-5", "−5 degrees", "minus 5", "0", "00", "007", "0.0", "1.0", "2.50", "10.000", "1,000.00", "1,23,456",
            "12,34", "1,2345", "123,456,789,012", "3.14159265358979323846", "0.1", "0.0001", "0.00001", "1.5e10",
            "1e-5", "6.02e23", "99999", "100000", "1000000000000", "12345678901234567890", "000000123456",
            "1st 2nd 3rd 4th 11th 12th 13th 21st 22nd 23rd 101st 111th 1000th 1000001st",
            "1001 2001 2009 2010 1900 1999 1066 2525 2999 3000 1000", "4x4", "Il-76", "and/or", "pay-per-view",
            "24/7", "9/11", "50/50", "1/2", "3/4 cup", "1920x1080", "a 2-for-1 deal", "1990s", "the '90s", "80's",
            "#1", "No. 5", "5,000-strong", "$5-10", "$3,000, 20%", "£1.50", "€99.99", "$1.2.3", "$.5", "$0.5",
            "$100.", "$ 100", "$100 million", "$1.5 billion", "100%", "100 %", "3.5%", "1,000%", "5,%",
            "1,2.3.4%", "x5th", "5thx", "the 5th.", "2nd-hand", "10am", "10 a.m.", "3 p.m.", "i.e. this", "e.g. that",
            "v1.2.3", "1.2.3.4", "192.168.0.1", "3:30", "12:00", "2023-10-22", "10/22/2023", "22.10.2023",
            "Room 101", "Apollo 11", "Boeing 747-400", "F-16", "B-52s", "COVID-19", "MP3", "4G", "5G LTE", "H2O", "CO2",
            "R2-D2", "C-3PO", "007 Bond", "1.", "1..2", "1...3", "12.", ".5", ".", "5.", "1,", ",5", "1,,000",
            "٣٤٥٦٧", "١٢٣", "٣rd", "١,٢٣٤", "٣.١٤", "१२३४५", "１２３", "1２3", "²", "½", "Ⅻ", "x²", "10⁶", "٣-٤", "٣x٤", "$٣"]
    # error cases (Python raises): huge numbers, invalid int literals, inflect index errors
    out += ["9" * 103, "1" * 102, "1" + "0" * 102, "$,5", "$.,5", "abc" + "1" * 40, "x" + "7" * 37 + "y",
            "1" * 4301, "1" * 310 + ".5", "0." + "0" * 330 + "1", "1" * 30 + ".5", "1" * 120 + "th",
            "1," + ",".join(["000"] * 40), "$" + "9" * 50, "abc" + "9" * 36]
    return out


def gen_misc_cases(r, homographs):
    out = []
    acr = ["BBC", "NASA", "IOC", "FBI", "FBIs", "UNESCO", "ABCDEF", "USAs", "IT", "OK", "UK", "US", "USA", "NATO",
           "CEOs", "PhD", "DNA", "HTML", "AIDS", "ASAP", "AAs", "Ss", "SS", "NYC's", "IBM", "TV", "TVs", "GPU",
           "LOL", "ZZZZ", "QWERTY", "XYZs", "ABCs", "ATMs"]
    for a in acr:
        out += [f"The {a} said so.", f"{a}", f"{a.lower()} and {a}", f"({a})"]
    for w, (p1, p2, pos) in homographs.items():
        a, b, c = r.choice(WORDS), r.choice(WORDS), r.choice(WORDS)
        out.append(f"I {w} the {a}.")
        out.append(f"The {w} {b} {c}.")
        if r.random() < 0.5:
            out.append(f"{w.capitalize()}, they said, {a} {w} {b}!")
    for _ in range(150):
        out.append(" ".join(r.choice(NAMES + [rand_word(r).capitalize() for _ in range(3)]) for _ in range(r.randint(1, 4))) +
                   r.choice([".", "?", "!", "", ","]))
    for _ in range(120):
        out.append(" ".join(rand_word(r) for _ in range(r.randint(1, 8))))
    out += ["pay-per-view", "low-tech", "state-of-the-art", "well-known author", "mother-in-law's", "e-mail",
            "x-ray", "T-shirt", "re-enter", "co-operate", "un-American", "self-esteem", "--", "a--b", "a - b",
            "a ‐ b", "a‐b", "well‐known", "a/b/c", "http://www.example.com/path", "www.google.com", "mr.smith",
            "end.next", "john@example.com", "@user #tag", "e.g.", "i.e.", "etc.", "Dr. Smith", "Mr. and Mrs. Jones",
            "U.S.A.", "a.b.c.d.e", "a.bc", "abc.defghijklmnopq", "x.y-z.co", "one-two.three", "don't", "I'll", "we'd",
            "you're", "they've", "can't", "won't", "o'clock", "rock'n'roll", "ma'am", "y'all", "'tis", "goin'",
            "the dogs' bones", "James's", "it's", "its", "'quoted'", "\"double\"", "“smart quotes”", "‘single’",
            "`backtick`", "He said, \"Hi!\"", "«guillemets»", "¿Qué?", "¡Hola!", "café", "naïve", "Zoë", "résumé",
            "São Paulo", "Ångström", "Kraków", "Dvořák", "Øresund", "Straße", "ﬁnance", "Æsop", "œuvre", "İstanbul",
            "ıi", "ſtrong", "Kelvin", "Ω", "µs", "Ä Ö Ü ä ö ü ß", "naïve café — résumé…", "wait… what?", "yes—no",
            "1990—1995", "A–Z", "ok… ok…", "hello 👋 world", "🙂", "I ❤ NY", "😂😂😂", "日本語のテキスト", "中文 English mix",
            "Привет мир", "Γειά σου", "مرحبا بالعالم", "שלום", "नमस्ते", "한국어", "ไทย", "Tiếng Việt", "hello\tworld",
            "line1\nline2", "a b", "a b", "a　b", "x​y", "a b", "zero‍width", "﻿bom",
            "", " ", "   ", "\n", "\t\n ", ".", "...", "!!!", "?!", "?", ",", "-", "—", "–", "…", "''", "\"\"", "'",
            "[]", "[ ]", "[", "]", "][", "!!!!!!!!", "........", ". . . .", "a . . . b", "????", "----", "aaaaaaaa",
            "hmmmmmm", "sooooo good", "nooooo!!!!", "a,b,c", "a;b:c", "semi; colon: test", "(parenthetical)",
            "[bracketed text]", "{curly}", "<angle>", "a & b", "a + b = c", "50 * 3", "x^2", "100°C", "98.6°F",
            "~approx", "back\\slash", "pipe|char", "under_score", "snake_case_name", "CamelCaseName", "iPhone",
            "McDonald's", "O'Neil", "AT&T", "Q&A", "R&D", "B2B", "P2P", "COVID", "Covid", "covid", "ALL CAPS SENTENCE HERE",
            "Mixed CASE words", "THE END.", "I", "A", "a", "I am.", "Read it.", "He read it.", "lead the way",
            "lead poisoning", "live music", "we live here", "a record", "to record", "close the door", "close by",
            "tear in my eye", "tear the paper", "wind blows", "wind the clock", "bass guitar", "bass fish",
            "present the present", "object to the object", "refuse the refuse", "the wound was wound"]
    ev = ["[laugh]", "Hello [laugh] world", "[ dog ]", "[foo] bar", "[LAUGH]", "[laugh][cough]", "[pause]",
          "[reserved3]", "[laugh", "a [b c] d", "[ laugh ] and [ CRY ]", "[dog] barks [cat] meows [bird]",
          "[applause]!", "Thanks [applause]. Bye [laugh]", "[_]", "[laugh_]", "[ ]x[ ]", "[[laugh]]",
          "[laugh]]]", "[[[laugh]", "x[laugh]y", "1[laugh]2", "$5 [cough] 10%", "[knock] [knock] who's there?",
          "[sigh] I don't know.", "[thunder] BBC news [clock]", "[footsteps]—then silence…", "[typing] 1,234 words",
          "[laugh1]", "[la ugh]", "[laugh 2]", "[yawn][sneeze][sniff][throat][breath][scream][yell][cheer]",
          "[phone] [bell] [horn] [glass] [siren] [door]", "[reserved0] [reserved12] [reserved13]", "[Laugh]ing",
          "[Kknock]", "[ſigh]", "[laugh]   [laugh]", "  [laugh]  ", "[laugh].", "[laugh],"]
    out += ev
    return out


def gen_fuzz(r, pool_words, n):
    chars = list("abcxyzABCXYZ0123456789 ,.;:!?-'\"/()[]$£€%&*_#@~") + ["—", "–", "…", "“", "”", "‘", "’", "é",
                                                                         "ñ", "ü", "日", "١", "٣", " ", "×", "‐", "`"]
    out = []
    for _ in range(n):
        parts = []
        for _ in range(r.randint(1, 12)):
            x = r.random()
            if x < 0.4:
                parts.append(r.choice(pool_words))
            elif x < 0.6:
                parts.append(str(r.randint(0, 10 ** r.randint(1, 8))))
            elif x < 0.75:
                parts.append(r.choice(["[laugh]", "[dog]", "[foo]", "km/h", "$5", "20%", "1st", "BBC", "e.g.",
                                       "1,000", "3.5", "-", "—", "...", "'s", "n't", "x", "4x4", "1990-1995"]))
            else:
                parts.append("".join(r.choice(chars) for _ in range(r.randint(1, 5))))
        sep = r.choice([" ", "", " ", ", ", "-"])
        out.append(sep.join(parts))
    return out


def gen_markup(r, en, tags):
    style_tags = [t for t in tags if t not in ("slow", "fast", "loud")]
    pros = ["emph", "slow", "fast", "high", "low", "loud", "soft"]
    out = []
    fixed = [
        "Hello *world*.", "This is *very important* news.", "*Hi* there *you*!", "**", "*a*b*c*", "* spaced *",
        "*multi\nline*", "Wait [pause 300ms] go.", "Wait [pause 1.5s] go.", "Wait [pause] go.", "[pause 0] x",
        "[pause 99999] x", "[PAUSE 2S] x", "[Pause 250 MS] x", "[pause 2 s]", "[pause .5s] a", "[pause 5.] b",
        "[pause 1.2.3] err", "[pause .] err", "[pause 7s]", "[pause 4999ms] [pause 5001ms]", "[pause 1e3] x",
        "<slow>slowly now</slow> normal", "<fast>quick</fast> <high>up</high> <low>down</low>",
        "<loud>LOUD</loud> <soft>soft</soft>", "<slow><high>both</high></slow> none",
        "<slow><slow>twice</slow> once</slow> zero", "<emph>explicit emph</emph> word", "<slow>unclosed slow text",
        "</slow>closing only", "<whisper>a secret</whisper> said aloud.", "<sadness>so sad.</sadness> <excited>Yay!</excited>",
        "<whisper>one. two. three.</whisper> four.", "<whisper>outer <sadness>inner</sadness> outer</whisper> out",
        "<foo>unknown tag</foo> text", "<style prompt=\"angry and fast\">err</style>", "<style prompt=\"\">empty prompt</style> x",
        "<style>no prompt</style> x", "</style>dangling", "<none>none tag</none>", "<WHISPER>caps</WHISPER>",
        "< whisper >spaced</ whisper >", "<whisper prompt=\"x\">prompt ignored</whisper>", "<emph prompt=\"y\">p</emph>",
        "A. B! C? D", "First sentence. Second sentence!  Third?", "No split.Here", "Mr. Smith went. Done.",
        "Ellipsis... next", "Stop. PAUSE", "Stop. BREAK go", "Stop. EVENT", "PAUSE", "BREAK", "EVENT",
        "<slow>PAUSE</slow>", "x <whisper>EVENT</whisper> y", "Hello [laugh] world.", "[dog] barks. [cat]",
        "*[laugh]*", "<slow>[laugh] haha</slow>", "[foo] unknown", "[pauſe 3] x", "<ſlow>x</ſlow>", "<Koud>k</Koud>",
        "[pause 3 ſ] x", "", " ", "...", "*", "<>", "< >", "<slow", "[pause", "*emph without close",
        "1990-1995 *was* great. $5 <slow>million</slow>.", "<slow>The BBC</slow> said *NASA* won.",
        "<high>pay-per-view</high> 4x4", "*don't* stop", "*1,000* people", "<fast>3.5%</fast> rate",
        "<sarcasm>Oh *great*.</sarcasm> Thanks. <tired>[sigh] fine.</tired>",
        "<emph>a</emph><emph>b</emph> c", "w1 <slow>w2 w3</slow> w4 <fast>w5</fast>.",
        "<slow>a <fast>b</slow> c</fast> d", "<whisper>a <slow>b</whisper> c</slow> d",
        "<whisper>unclosed span. Next sentence.", "text </whisper> more", "<slow>  </slow> x", "x <slow></slow> y",
        "<whisper> </whisper> x", "a　b c. d e", "<loud>[pause 200ms]</loud> x",
        "1" * 120 + " big", "<slow>" + "9" * 110 + "</slow>",
        "*a* " * 300, "<slow>" * 1000 + "x" + "</slow>" * 1000, "<whisper>" * 500 + "deep" + "</whisper>" * 499,
        "[pause 10ms]" * 500, "w. " * 1000,
    ]
    for t in fixed:
        out.append((t, None, None))
    for t in fixed[:40]:
        out.append((t, r.choice(["whisper", "bogus", "none", "", "sadness"]), r.choice([None, {"rate": 1.2, "loud": -0.5},
                                                                                       {"pause": 0.0}, {"z": 1.0, "a": 2.5}])))
    sents = [s for s in en if len(s) < 140]
    for _ in range(470):
        k = r.randint(1, 3)
        txt = " ".join(r.choice(sents) for _ in range(k))
        words = txt.split(" ")
        ops = r.randint(1, 4)
        for _ in range(ops):
            if len(words) < 2:
                break
            i = r.randrange(len(words))
            j = min(len(words), i + r.randint(1, 4))
            x = r.random()
            if x < 0.25:
                words[i] = "*" + words[i] + "*"
            elif x < 0.5:
                p = r.choice(pros)
                words[i] = f"<{p}>" + words[i]
                if r.random() < 0.9:
                    words[j - 1] = words[j - 1] + f"</{p}>"
            elif x < 0.7:
                t = r.choice(style_tags + ["foo", "style"])
                words[i] = f"<{t}>" + words[i]
                if r.random() < 0.85:
                    words[j - 1] = words[j - 1] + f"</{t}>"
            elif x < 0.85:
                words.insert(i, r.choice(["[pause]", "[pause 300ms]", "[pause 1.5s]", "[pause 2s]", "[pause 120]",
                                          "[pause 0.25 s]"]))
            else:
                words.insert(i, r.choice(["[laugh]", "[cough]", "[dog]", "[foo]", "[ sigh ]", "[knock]"]))
        tag = r.choice([None, None, None, "whisper", "sadness", "bogus", "none"])
        ctrl = r.choice([None, None, {"rate": 0.5}, {"loud": 1.0, "f0_med": -1.25}])
        out.append((" ".join(words), tag, ctrl))
    return out


def main():
    pkg, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    n_g2p_rand = int(sys.argv[sys.argv.index("--g2p-words") + 1]) if "--g2p-words" in sys.argv else 4000
    sys.path.insert(0, str(pkg))
    from steertts_edge import text as T
    T.set_assets(pkg / "front")
    from steertts_edge.markup import MarkupSynth
    cfg = json.loads((pkg / "config.json").read_text())
    tags = cfg["tags"]

    class Stub:
        tag2i = {t: i for i, t in enumerate(tags)}

    ms = MarkupSynth(Stub())
    meta = json.loads(lzma.decompress((pkg / "front" / "front.json.xz").read_bytes()))
    r = random.Random(20261022)
    en, zh = real_texts()
    cases = list(en) + list(zh)
    cases += gen_number_cases(r)
    cases += gen_misc_cases(r, meta["homographs"])
    pool = [w for s in en for w in s.split()] or WORDS
    cases += gen_fuzz(r, pool, 900)
    # long inputs
    for _ in range(12):
        cases.append(" ".join(r.choice(en) for _ in range(r.randint(20, 60))))
    cases.append(" ".join(["word"] * 3000))
    cases.append("a." * 2000)
    cases.append("[laugh] " * 300)
    # adversarial inputs (deep regex repetition, huge words / numbers)
    cases += ["a" * 20000, "abc'" * 3000, "1," + ",".join(["000"] * 5000), "1," + ",".join(["000"] * 1000) + ".5",
              ". " * 5000, "a-" * 5000, "a." * 5000, "[laugh]" * 2000, "x" * 5000 + "!", "9" * 5000,
              "$" + "1," * 3000 + "5", "-" * 10000, "'" * 10000, "a " * 10000]

    def err(e):
        return f"{type(e).__name__}: {e}"

    out_dir.mkdir(parents=True, exist_ok=True)
    n_ph = n_mk = 0
    with open(out_dir / "golden.jsonl", "w", encoding="utf-8") as f:
        for t in cases:
            rec = {"k": "ph", "text": t}
            try:
                rec["norm"] = T.normalize(t)
            except Exception as e:  # noqa: BLE001
                rec["norm"] = None
                rec["norm_error"] = err(e)
            try:
                ids, widx, _ = T.to_phonemes(t)
                rec["ids"], rec["widx"] = ids, widx
            except Exception as e:  # noqa: BLE001
                rec["error"] = err(e)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_ph += 1
        for t, tag, ctrl in gen_markup(r, en, tags):
            rec = {"k": "mk", "text": t, "tag": tag, "ctrl": [[k, v] for k, v in ctrl.items()] if ctrl else None}
            try:
                segs = ms.parse(t, tag=tag, ctrl=ctrl)
                ss = []
                for s in segs:
                    ws = []
                    for w, sh in sorted(s.word_shift.items()):
                        ws.append([w, sh.get("pitch", 0.0), sh.get("energy", 0.0), sh.get("dur", 1.0)])
                    ss.append({"text": s.text, "tag": s.tag, "ctrl": [[k, v] for k, v in s.ctrl.items()], "ws": ws,
                               "pause_ms": s.pause_ms if isinstance(s.pause_ms, int) else -1,
                               "event": s.event if isinstance(s.event, str) else "<style-tuple>"})
                rec["segs"] = ss
            except Exception as e:  # noqa: BLE001
                rec["error"] = err(e)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_mk += 1
    # GRU G2P words: every member-only lexicon word + random pseudo-words
    from steertts_edge.edge_front.lexicon import Lexicon
    from steertts_edge.edge_front.gru_np import GruG2P, PHONEMES
    g = GruG2P(pkg / "front" / "gru.npz")
    p2i = {p: i for i, p in enumerate(PHONEMES)}
    words = [w for w, p in Lexicon(pkg / "front" / "cmudict.lex").items() if not p]
    words += sorted({rand_word(r) for _ in range(n_g2p_rand)})
    words += ["", "a", "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzz", "end.next", "don't", "a-b", "x'y.z", "q"]
    with open(out_dir / "golden_g2p.jsonl", "w", encoding="utf-8") as f:
        for w in words:
            ph, gap = g.predict(w, margin=True)
            f.write(json.dumps({"w": w, "p": [p2i[p] for p in ph], "gap": round(float(gap), 6)}) + "\n")
    print(f"golden: {n_ph} to_phonemes cases, {n_mk} markup cases, {len(words)} g2p words -> {out_dir}")


if __name__ == "__main__":
    main()
