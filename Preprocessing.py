import re
import unicodedata

def normalize_text(text):
    text = text.lower()
    text = "".join(char for char in text if not unicodedata.category(char).startswith("P"))
    text = re.sub(r"\s+", " ", text).strip()
    return text