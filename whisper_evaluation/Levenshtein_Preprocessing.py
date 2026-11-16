# -*- coding: iso-8859-1 -*-

from whisper_evaluation.Regex_Preprocessing import correct_numbers, remove_punctuation
from rapidfuzz.distance import Levenshtein
import re

def normalize_text(text):
    text = text.lower()
    text = remove_punctuation(text)
    text = correct_numbers(text)
    text = replace_false_joins(text)
    vocab = ["zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun",
            "eo", "ba", "ndb", "moly", "mo",
            "ly", "klein", "mittel", "gross", "mag", "lyklein", "lymittel", "lygross", "lymag",
            "het", "tox", "stab", "met", "hettox", "hetstab", "hetmet"]
    text = replace_with_closest_match(text, vocab)
    text = replace_false_splits(text)

    return text

def get_closest_match(token, vocab):
    return min((word for word in vocab), key=lambda word: Levenshtein.distance(token, word))

def replace_with_closest_match(text, vocab):
    return " ".join(get_closest_match(token, vocab) for token in text.split())

def replace_false_splits(text):
    corrections = {
        "ly klein": "lyklein",
        "ly mittel": "lymittel",
        "ly gross": "lygross",
        "ly mag": "lymag",
        "het tox": "hettox",
        "het stab": "hetstab",
        "het met": "hetmet",
    }

    return re.sub(
        r"\b(?:ly klein|ly mittel|ly gross|ly mag|het tox|het stab|het met)\b",
        lambda m: corrections[m.group()],
        text
    )

def replace_false_joins(text):
    number_words = (
        "zwei", "drei", "vier", "fünf",
        "sechs", "sieben", "acht", "neun"
    )
    pattern = r"\b(" + "|".join(number_words) + r")\s*"

    return re.sub(pattern, r"\1 ", text).strip()


