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
OUTPUT_DIR = Path("//data/audio")
CSV_FILE = Path("//data/transcripts.csv")

NUMBERS = ["zwei","drei","vier"]

CATEGORIES = ["het", "ly", "mo", "eo", "ba", "ndb", "hettox", "hetstab", "hetmet", "lyklein",
              "lymittel", "lygross", "lymag", "moly"]

# GENERATE NORMAL PHRASE

def generate_item():
    category = random.choice(CATEGORIES)

    if random.random() < 0.40:
        number = random.choice(NUMBERS)
        return f"{number} {category}"

    return category


def generate_phrase(min_items=5, max_items=15):
    number_of_items = random.randint(min_items, max_items)
    items = []

    for _ in range(number_of_items):
        items.append(generate_item())

    return " ".join(items)


# GENERATE CONFUSION PHRASE
def generate_confusion_phrase():
    patterns = [["mo", "ly"], ["moly"], ["mo", "moly"], ["moly", "mo"], ["ly", "moly"], ["moly", "ly"],
                ["mo", "ly", "moly"], ["moly", "mo", "ly"], ["mo", "moly", "ly"], ["ly", "mo", "moly"],
                ["moly", "moly"], ["mo", "ly", "mo", "ly"], ["het", "hettox"], ["hettox", "het"],
                ["het", "hetstab"], ["hetstab", "het"], ["het", "hetmet"], ["hetmet", "het"],
                ["hettox", "hetstab"], ["hetstab", "hettox"], ["hettox", "hetmet"], ["hetmet", "hettox"],
                ["hetstab", "hetmet"], ["hetmet", "hetstab"], ["het", "hettox", "hetstab", "hetmet"],
                ["hetmet", "hetstab", "hettox", "het"], ["het", "het", "hettox"], ["het", "hettox", "het"],
                ["hettox", "het", "hettox"], ["het", "hetmet", "het"], ["hetmet", "het", "hetmet"],
                ["het", "hetstab", "het"], ["hetstab", "het", "hetstab"], ["het", "het", "het"],
                ["hettox", "hettox"], ["hetstab", "hetstab"], ["hetmet", "hetmet"], ["ly", "lyklein"],
                ["lyklein", "ly"], ["ly", "lymittel"], ["lymittel", "ly"], ["ly", "lygross"],
                ["lygross", "ly"], ["ly", "lymag"], ["lymag", "ly"], ["lyklein", "lymittel"],
                ["lymittel", "lyklein"], ["lyklein", "lygross"], ["lygross", "lyklein"], ["lyklein", "lymag"],
                ["lymag", "lyklein"], ["lymittel", "lygross"], ["lygross", "lymittel"], ["lymittel", "lymag"],
                ["lymag", "lymittel"], ["lygross", "lymag"], ["lymag", "lygross"],
                ["ly", "lyklein", "lymittel"], ["ly", "lygross", "lymag"],
                ["ly", "lyklein", "lymittel", "lygross", "lymag"],
                ["lymag", "lygross", "lymittel", "lyklein", "ly"], ["ly", "ly", "lyklein"],
                ["ly", "lyklein", "ly"], ["lyklein", "ly", "lyklein"], ["ly", "lymittel", "ly"],
                ["lymittel", "ly", "lymittel"], ["ly", "lygross", "ly"], ["lygross", "ly", "lygross"],
                ["ly", "lymag", "ly"], ["lymag", "ly", "lymag"], ["ly", "ly"], ["lyklein", "lyklein"],
                ["lymittel", "lymittel"], ["lygross", "lygross"], ["lymag", "lymag"]]

    special = random.choice(patterns)
    before = [generate_item() for _ in range(random.randint(1, 4))]
    after = [generate_item() for _ in range(random.randint(1, 4))]
    sequence = before + special + after

    return " ".join(sequence)


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
            writer.writerow(["file", "text"])


def get_next_filename():
    existing = list(OUTPUT_DIR.glob("recording_*.wav"))
    number = len(existing) + 1
    return f"recording_{number}.wav"


def save_transcript(filename, transcript):
    with open(CSV_FILE, "a", newline="", encoding="ISO-8859-1") as f:
        writer = csv.writer(f)
        writer.writerow([filename, transcript])


# RECORD TRAINING DATA

def record_audio():
    print()
    print("Recording...")
    print("Press ENTER when finished.")
    print()

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

    print(sd.query_devices())
    print()
    print("Default input/output:")
    print(sd.default.device)

    while True:

        phrase = get_training_phrase()

        print()
        print("--------------------------------")
        print()
        print("READ:")
        print()
        print(phrase)
        print()
        print("--------------------------------")
        print()

        command = input(
            "[ENTER] record | "
            "[S] skip | "
            "[Q] quit: "
        ).strip().lower()

        if command == "q":
            break

        if command == "s":
            continue

        print()
        print("Press ENTER after speaking.")

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
            phrase
        )

        print()
        print(f"Saved: {filename}")
        print(f"Text:  {phrase}")


if __name__ == "__main__":
    main()