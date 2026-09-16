# -*- coding: iso-8859-1 -*-

from pathlib import Path
import csv
import re

from faster_whisper import WhisperModel
from rapidfuzz.distance import Levenshtein
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "data" / "transcripts.csv"
AUDIO_DIR = BASE_DIR / "data" / "audio"

PERFORMANCE_FILE = BASE_DIR / "data" / "results" / "model_performance.csv"
MODEL_SIZE = "large-v3"

OUTPUT_FILE = BASE_DIR / "data" / "results" / f"whisper_{MODEL_SIZE}.csv"

model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")

def normalize_text(text : str) -> str:
    text = text.lower()
    text = re.sub(r"[^\wäöüß]+", " ", text)
    text = re.sub(r"ß", "ss", text)
    text = re.sub(r"gros", "gross", text)

    text = re.sub(r"(lu|lü|lüh|lüge|li)","ly", text)
    text = re.sub(r"(mark|marc|mak|mac)","mag", text)
    text = re.sub(r"(mat|matt|mett)","met", text)
    text = re.sub(r"bar", "ba", text)
    text = re.sub(r"ewe", "eo", text)

    text = re.sub(r"(toks|tocks|talks|hox)","tox", text)
    text = re.sub(r"\box", "tox", text)

    text = re.sub(r"(hätt|hät|hat|hand|head|hed|hett)","het", text)

    text = re.sub(r"n d b", "ndb", text)

    text = re.sub(r"2", "zwei", text)
    text = re.sub(r"3", "drei", text)
    text = re.sub(r"4", "vier", text)
    text = re.sub(r"5", "fünf", text)
    text = re.sub(r"6", "sechs", text)
    text = re.sub(r"7", "sieben", text)
    text = re.sub(r"8", "acht", text)
    text = re.sub(r"9", "neun", text)

    text = re.sub(r"\b(zwei|drei|vier|fünf|sechs|sieben|acht|neun)(?=[a-zäöüß])",r"\1 ", text)
    text = re.sub(r"(?<=[a-zäöüß])(zwei|drei|vier|fünf|sechs|sieben|acht|neun)\b", r" \1", text)

    text = re.sub(r"ly mag", "lymag", text)
    text = re.sub(r"ly gross", "lygross", text)
    text = re.sub(r"ly klein", "lyklein", text)
    text = re.sub(r"ly mittel", "lymittel", text)
    text = re.sub(r"het met", "hetmet", text)
    text = re.sub(r"het stab", "hetstab", text)
    text = re.sub(r"het tox", "hettox", text)
    text = re.sub(r"hetox", "hettox", text)
    return text.strip()

def get_metrics(reference : str, prediction : str) -> tuple[int, int]:
    reference_tokens = reference.split()
    prediction_tokens = prediction.split()
    errors = Levenshtein.distance(reference_tokens,prediction_tokens)
    tokens = len(reference_tokens)
    #token_error_rate = (errors / len(reference_tokens) if reference_tokens else 0)
    return errors, tokens#, token_error_rate

def get_recording_number(filename : str) -> int:
    return int(filename.removeprefix("recording_").removesuffix(".wav"))

with open(CSV_FILE, "r", encoding="ISO-8859-1") as f:
    references = [row for row in csv.DictReader(f)]

results = []
total_errors = 0
total_reference_tokens = 0

for index, row in enumerate(references, start=1):
    filename = row["file"]
    speaker = row["speaker"]
    reference = normalize_text(row["text"])
    audio_path = AUDIO_DIR / filename

    if audio_path.exists():
        print(f"[{index}/{len(references)}] {filename}")

        segments, info = model.transcribe(str(audio_path), language="de", task="transcribe",
                                          beam_size=5, condition_on_previous_text=False,)

        raw_text = " ".join(segment.text.strip() for segment in segments).lower()
        rawerrors, rawtokens = get_metrics(reference, raw_text)

        prediction = normalize_text(raw_text)
        errors, tokens = get_metrics(reference, prediction)
        total_errors += errors
        total_reference_tokens += tokens

        results.append({"file": filename, "speaker": speaker, "reference": reference,
                        "whisper_raw": raw_text, "whisper_normalized": prediction,
                        "errors_raw": rawerrors, "errors_normalized": errors})

    else:
        print(f"[{index}/{len(references)}] {filename} skipped. File missing or corrupted.")


with open(OUTPUT_FILE, "w", newline="", encoding="ISO-8859-1") as f:
    fieldnames = ["file", "speaker", "reference", "whisper_raw", "whisper_normalized", "errors_raw",
                  "errors_normalized"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

overall_ter = total_errors / total_reference_tokens

performance_row = {"timestamp": datetime.now().isoformat(timespec="seconds"),
                   "model": f"whisper_{MODEL_SIZE}", "files": len(results),
                   "reference_tokens": total_reference_tokens, "errors": total_errors,
                   "ter": overall_ter,}


with open(PERFORMANCE_FILE, "a", newline="", encoding="ISO-8859-1") as f:
    fieldnames = ["timestamp", "model", "files", "reference_tokens", "errors", "ter"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)

    writer.writerow(performance_row)