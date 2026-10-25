"""Text -> ARPAbet phoneme ids with word indices (g2p_en)."""
import re
from functools import lru_cache

_PUNCT = list(",.?!;:-")
_ARPA = ["AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY", "F", "G", "HH", "IH", "IY",
         "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH"]
_VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
SYMBOLS = ["<pad>", "<sp>"] + _PUNCT + [p + s for p in _ARPA for s in (["0", "1", "2"] if p in _VOWELS else [""])]
N_BASE = len(SYMBOLS)  # phoneme + punctuation vocabulary (ids below the sound-event tokens)
# sound-event tokens, written inline as [laugh], [dog] ... : in-voice nonverbals first, then environmental events,
# then reserved slots so the vocabulary can grow without resizing the embedding again. Appended after the phonemes,
# so ids < N_BASE are unchanged and a model without event tokens still works (tokens it lacks are dropped).
NONVERBAL = ["laugh", "cry", "scream", "yell", "cheer", "cough", "sneeze", "yawn", "throat", "breath", "sigh", "sniff"]
EVENTS = ["dog", "cat", "bird", "knock", "door", "phone", "bell", "horn", "applause", "glass", "siren", "footsteps",
          "typing", "thunder", "clock"]
EVENT_TOKENS = NONVERBAL + EVENTS + [f"reserved{i}" for i in range(13)]
SYMBOLS = SYMBOLS + [f"[{e}]" for e in EVENT_TOKENS]
SYM2ID = {s: i for i, s in enumerate(SYMBOLS)}
PAD, SP = 0, 1
_EVENT_RE = re.compile(r"\[\s*([A-Za-z_]+)\s*\]")


def _ensure_nltk():
    """g2p_en needs NLTK's POS tagger + CMUdict. nltk>=3.9 looks the tagger up under its '_eng' name, which
    g2p_en's own first-run download does not fetch; get whichever is missing once (into ~/nltk_data)."""
    import nltk
    for res, name in (("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
                      ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
                      ("corpora/cmudict", "cmudict")):
        try:
            nltk.data.find(res)
        except LookupError:
            nltk.download(name, quiet=True)


@lru_cache(maxsize=1)
def _g2p():
    _ensure_nltk()
    from g2p_en import G2p
    return G2p()


_UNITS = {"km/h": "kilometers per hour", "km/hr": "kilometers per hour", "m/s": "meters per second", "mph": "miles per hour",
          "km": "kilometers", "kg": "kilograms", "cm": "centimeters", "mm": "millimeters", "ft": "feet", "lbs": "pounds",
          "lb": "pounds", "kHz": "kilohertz", "Hz": "hertz", "GB": "gigabytes", "MB": "megabytes"}
_CURRENCY = {"$": "dollars", "£": "pounds", "€": "euros"}


def _num(s, to="cardinal"):
    from num2words import num2words
    return num2words(s, to=to).replace("-", " ").replace(",", "")


def _expand(text: str) -> str:
    """Spell out what g2p_en gets wrong (CV3-Eval en: 0.6 + 0.6 WER points of 3.3 came from these).
    g2p_en looks words up in CMUdict and sends anything else to a neural guesser - including every hyphenated
    compound ("pay-per-view" -> P EY V ER P IY, "low-tech" -> L OW T CH EH) and "4x4" / "Il-76" / "km/h". It only
    expands bare integers, reading "2,243" as "two" + "two hundred forty three"."""
    text = re.sub(r"(\d)\s*[-–]\s*(\d)", r"\1 to \2", text)                     # ranges: 1990-1995, 10-20
    text = re.sub(r"(\d+)\s*[x×]\s*(\d+)", r"\1 by \2", text)                   # 4x4, 1920x1080
    text = re.sub(r"(?<=[A-Za-z0-9])[-‐/](?=[A-Za-z0-9])", " ", text)           # compounds: pay-per-view, Il-76, and/or
    # the amount ends on a digit (a trailing comma is punctuation), and the space after it is only consumed together
    # with a following "million/billion/thousand" (otherwise "$3,000, 20%" read "three thousand, dollars20")
    text = re.sub(r"([$£€])\s?(\d(?:[\d,]*\d)?(?:\.\d+)?)(?:\s*(million|billion|thousand)\b)?",
                  lambda m: f"{m.group(2)} {m.group(3) + ' ' if m.group(3) else ''}{_CURRENCY[m.group(1)]}", text)
    text = re.sub(r"(\d[\d,]*(?:\.\d+)?)\s?%", r"\1 percent", text)
    for u in sorted(_UNITS, key=len, reverse=True):                            # 165 km h (after the / split) / 165 km/h
        pat = re.escape(u).replace("/", "[ /]")
        text = re.sub(rf"(\d)\s?{pat}\b", rf"\1 {_UNITS[u]}", text)
    text = re.sub(r"\b(\d+)(st|nd|rd|th)\b", lambda m: _num(int(m.group(1)), "ordinal"), text)
    text = re.sub(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b", lambda m: _num(float(m.group(0).replace(",", "")))
                  if "." in m.group(0) else _num(int(m.group(0).replace(",", ""))), text)
    text = re.sub(r"\b\d+\.\d+\b", lambda m: _num(float(m.group(0))), text)
    text = re.sub(r"\b\d{5,}\b", lambda m: _num(int(m.group(0))), text)         # g2p_en reads long digit runs badly
    text = re.sub(r"(?<=[a-z])([A-Z]{2,5}s?)\b", r" \1", text)   # a word ending in capitals: SteerTTS -> Steer TTS, macOS -> mac OS
    cmu = _g2p().cmu  # acronyms CMUdict does not know (IOC, BBC): letter by letter; NASA, UNESCO keep their entry
    text = re.sub(r"\b[A-Z]{2,5}s?\b", lambda m: m.group(0) if m.group(0).lower() in cmu
                  else " ".join(m.group(0).rstrip("s")) + ("'s" if m.group(0).endswith("s") and len(m.group(0)) > 2 else ""), text)
    return text


def normalize(text: str) -> str:
    text = re.sub(r"[\"“”‘’`]", lambda m: "'" if m.group(0) in "‘’`" else "", text)
    text = _expand(text)  # first: it reads digit-dash-digit as a range before dashes become pauses
    text = text.replace("—", ", ").replace("–", ", ").replace("…", ". ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def to_phonemes(text: str):
    """Returns (ids, word_idx, words). Word boundaries become <sp> tokens with word_idx -1;
    punctuation tokens get word_idx -1 as well. A trailing <sp> pads the end. Inline sound events ([laugh], [dog])
    become one token each, between <sp> tokens, with word_idx -1; unknown [...] markers are dropped."""
    text = normalize(text)
    if "[" in text:
        parts = _EVENT_RE.split(text)  # text, name, text, name, ...
        if len(parts) > 1:
            ids, widx, words, nw = [SP], [-1], [], 0
            for k, p in enumerate(parts):
                if k % 2:
                    tok = f"[{p.lower()}]"
                    if tok in SYM2ID:
                        ids += [SYM2ID[tok], SP]; widx += [-1, -1]
                elif p.strip(" []"):
                    i2, w2, wd = to_phonemes(p.replace("[", " ").replace("]", " "))
                    ids += i2[1:]; widx += [w + nw if w >= 0 else -1 for w in w2[1:]]
                    words += wd; nw = max([w for w in widx if w >= 0], default=-1) + 1
            return ids, widx, words
    raw = _g2p()(text)
    ids, widx, words = [SP], [-1], []
    cur, wi = [], -1
    for tok in raw + [" "]:
        if tok == " ":
            if cur:
                wi += 1
                ids += [SYM2ID[p] for p in cur]
                widx += [wi] * len(cur)
                cur = []
                ids.append(SP); widx.append(-1)
            continue
        if tok in SYM2ID and tok not in ("<pad>", "<sp>"):
            if tok in _PUNCT:
                if cur:
                    wi += 1
                    ids += [SYM2ID[p] for p in cur]
                    widx += [wi] * len(cur)
                    cur = []
                ids.append(SYM2ID[tok]); widx.append(-1)
            else:
                cur.append(tok)
    # words list re-derived from the normalised text for display (best effort alignment)
    words = re.findall(r"[A-Za-z0-9']+", text)
    return ids, widx, words
