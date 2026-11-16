# -*- coding: iso-8859-1 -*-
import re
import unicodedata

def normalize_text(text):
    text = text.lower()
    text = remove_punctuation(text)

    text = correct_numbers(text)
    text = correct_ba_eo(text)
    text = correct_moly(text)
    text = correct_ly(text)
    text = correct_het(text)

    return " ".join(text.split())

def remove_punctuation(text):
    return "".join(
        " " if unicodedata.category(char).startswith("P") else char
        for char in text
    )

def correct_numbers(text):
    numbers = {
        "2": "zwei",
        "3": "drei",
        "4": "vier",
        "5": "fünf",
        "6": "sechs",
        "7": "sieben",
        "8": "acht",
        "9": "neun",
    }

    return re.sub(
        r"\b[2-9]\b",
        lambda m: numbers[m.group()],
        text
    )


def correct_ba_eo(text):
    corrections = {
        "bar": "ba",
        "eu": "eo",
        "eho": "eo",
    }

    return re.sub(
        r"\b(?:bar|eu|eho)\b",
        lambda m: corrections[m.group()],
        text
    )


def correct_moly(text):
    # Only single-word variants; never merge "mo ly" or "mo lu"
    return re.sub(
        r"\b(?:molu|moli|molü|molli|molie)\b",
        "moly",
        text
    )


def correct_ly(text):
    # Full-word spelling variants
    corrections = {
        "kline": "klein",
        "cline": "klein",
        "mitel": "mittel",
        "mitl": "mittel",
        "mittl": "mittel",
        "gros": "gross",
        "gross": "gross",
        "groß": "gross",
        "grose": "gross",
        "mark": "mag",
        "mak": "mag",
        "mac": "mag",
        "mack": "mag",
    }

    pattern = (
        r"\b(?:ly|lü|lu)"
        r"(klein|kline|cline|mittel|mitel|mitl|mittl|"
        r"gros|gross|groß|grose|mag|mark|mak|mac|mack)\b"
        r"|\b(?:lü|lu)\b"
    )

    def replace(match):
        suffix = match.group(1)

        if suffix:
            return "ly" + corrections.get(suffix, suffix)

        return "ly"

    return re.sub(pattern, replace, text)


def correct_het(text):
    prefixes = r"(?:het|hett|hed|head|hät)"

    suffixes = {
        "stab": "stab",
        "stap": "stab",
        "ztab": "stab",
        "schtab": "stab",
        "met": "met",
        "mat": "met",
        "mett": "met",
        "tox": "tox",
        "toks": "tox",
        "tops": "tox",
        "dox": "tox",
        "docs": "tox",
        "ox": "tox",
    }

    pattern = (
        rf"\b{prefixes}"
        r"(stab|stap|ztab|schtab|met|mat|mett|"
        r"tox|toks|tops|dox|docs|ox)\b"
        rf"|\b{prefixes}\b"
    )

    def replace(match):
        suffix = match.group(1)

        if suffix:
            return "het" + suffixes[suffix]

        return "het"

    return re.sub(pattern, replace, text)
