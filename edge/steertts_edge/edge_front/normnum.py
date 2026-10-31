"""g2p_en.expand.normalize_numbers (keithito/tacotron numbers.py) on top of numwords.inflect_words (g2p_en front-end path)."""
import re

from .numwords import inflect_words

_comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
_decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
_pounds_re = re.compile(r"£([0-9\,]*[0-9]+)")
_dollars_re = re.compile(r"\$([0-9\.\,]*[0-9]+)")
_ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")
_number_re = re.compile(r"[0-9]+")


def _expand_dollars(m):
    match = m.group(1)
    parts = match.split(".")
    if len(parts) > 2:
        return match + " dollars"
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        return "%s %s, %s %s" % (dollars, "dollar" if dollars == 1 else "dollars", cents, "cent" if cents == 1 else "cents")
    if dollars:
        return "%s %s" % (dollars, "dollar" if dollars == 1 else "dollars")
    if cents:
        return "%s %s" % (cents, "cent" if cents == 1 else "cents")
    return "zero dollars"


def _expand_number(m):
    num = int(m.group(0))
    if 1000 < num < 3000:
        if num == 2000:
            return "two thousand"
        if 2000 < num < 2010:
            return "two thousand " + inflect_words(num % 100)
        if num % 100 == 0:
            return inflect_words(num // 100) + " hundred"
        return inflect_words(num, andword="", zero="oh", group=2).replace(", ", " ")
    return inflect_words(num, andword="")


def normalize_numbers(text):
    text = re.sub(_comma_number_re, lambda m: m.group(1).replace(",", ""), text)
    text = re.sub(_pounds_re, r"\1 pounds", text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_decimal_number_re, lambda m: m.group(1).replace(".", " point "), text)
    text = re.sub(_ordinal_re, lambda m: inflect_words(m.group(0)), text)
    text = re.sub(_number_re, _expand_number, text)
    return text
