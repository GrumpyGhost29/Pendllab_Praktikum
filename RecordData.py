import csv
import random
import time
from pathlib import Path
import sounddevice as sd
import soundfile as sf
import numpy as np

# Config
SAMPLE_RATE = 16000
CHANNELS = 1
BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "data" / "audio"
CSV_FILE = BASE_DIR / "data" / "transcripts.csv"

NUMBERS = ["zwei","drei", "vier", "fünf"]

CATEGORIES = ["het", "ly", "mo", "eo", "ba", "ndb", "hettox", "hetstab", "hetmet", "lyklein",
              "lymittel", "lygross", "lymag", "moly"]

# GENERATE NORMAL PHRASE

def generate_item():
    category = random.choice(CATEGORIES)

    if random.random() < 0.20:
        number = random.choice(NUMBERS)
        return f"{number} {category}"

    return category


def generate_phrase(min_items=5, max_items=15):
    number_of_items = random.randint(min_items, max_items)
    items = []

    for _ in range(number_of_items):
        items.append(generate_item())

    return " ".join(items)


MO_FAMILY = [
    "mo",
    "moly",
]

LY_FAMILY = [
    "ly",
    "lyklein",
    "lymittel",
    "lygross",
    "lymag",
]

HET_FAMILY = [
    "het",
    "hettox",
    "hetstab",
    "hetmet",
]


def generate_pair_patterns(group):
    patterns = []

    for first in group:
        for second in group:

            patterns.append([first, second])

    return patterns


def generate_cross_patterns(group_a, group_b):
    patterns = []

    for a in group_a:
        for b in group_b:

            patterns.extend([
                [a, b],
                [b, a],
                [a, b, a],
                [b, a, b],
            ])

    return patterns

CONFUSION_PATTERNS = []

# Within-family confusion
CONFUSION_PATTERNS += generate_pair_patterns(HET_FAMILY)
CONFUSION_PATTERNS += generate_pair_patterns(LY_FAMILY)
CONFUSION_PATTERNS += generate_pair_patterns(MO_FAMILY)

# Cross-family confusion
CONFUSION_PATTERNS += generate_cross_patterns(
    MO_FAMILY,
    LY_FAMILY
)

# Especially important manually specified cases
CONFUSION_PATTERNS += [
    ["mo", "ly"],
    ["moly"],

    ["mo", "ly", "moly"],
    ["moly", "mo", "ly"],

    ["mo", "ly", "mo", "ly"],
    ["moly", "moly"],

    ["mo", "moly", "ly"],
    ["ly", "mo", "moly"],
]

def generate_confusion_phrase():

    special = random.choice(CONFUSION_PATTERNS)

    before = [
        generate_item()
        for _ in range(random.randint(1, 3))
    ]

    after = [
        generate_item()
        for _ in range(random.randint(1, 3))
    ]

    return " ".join(before + special + after)


def get_training_phrase():
    if random.random() < 0.30:
        return generate_confusion_phrase()

    return generate_phrase()


def prepare_directories():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CSV_FILE.exists():
        CSV_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CSV_FILE, "w", newline="", encoding="ISO-8859-1") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "speaker", "text"])


def get_next_filename():
    existing = list(OUTPUT_DIR.glob("recording_*.wav"))
    number = len(existing) + 1
    return f"recording_{number:05d}.wav"


def save_transcript(filename, speaker_id, transcript):
    with open(CSV_FILE, "a", newline="", encoding="ISO-8859-1") as f:
        writer = csv.writer(f)
        writer.writerow([filename, speaker_id, transcript])


# RECORD TRAINING DATA

def record_audio():
    print("Recording...")
    print("Press ENTER when finished.")

    recording = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32")
    chunks = []
    def callback(indata, frames, time_info, status):
        if status:
            print(status)

        chunks.append(indata.copy())

    with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            callback=callback
    ):
        input()

    if len(chunks) == 0:
        print("WARNING: No audio was recorded.")
        return None

    audio = np.concatenate(chunks, axis=0)
    return audio


# MAIN

def main():

    prepare_directories()

    print()
    print("==============================")
    print(" Small STT Dataset Recorder")
    print("==============================")
    print()

    speaker_id = input(
        "Enter speaker ID: "
    ).strip()

    while not speaker_id:
        print("Speaker ID cannot be empty.")
        speaker_id = input(
            "Enter speaker ID: "
        ).strip()

    print()
    print(f"Speaker: {speaker_id}")
    print()

    print(sd.query_devices())
    print()
    print("Default input/output:")
    print(sd.default.device)

    while True:

        phrase = get_training_phrase()

        print()
        print()
        print()
        print("--------------------------------")
        print("READ:")
        print(phrase)
        print("--------------------------------")

        command = input(
            "[ENTER] record | "
            "[S] skip | "
            "[Q] quit: "
        ).strip().lower()

        if command == "q":
            break

        if command == "s":
            continue

        time.sleep(0.3)

        audio = record_audio()

        if audio is None:
            print("No audio captured. Recording skipped.")
            continue

        filename = get_next_filename()

        path = OUTPUT_DIR / filename

        sf.write(
            path,
            audio,
            SAMPLE_RATE,
            subtype="PCM_16"
        )

        save_transcript(
            filename,
            speaker_id,
            phrase
        )

        print()
        print(f"Saved: {filename}")
        print(f"Text:  {phrase}")


if __name__ == "__main__":
    main()