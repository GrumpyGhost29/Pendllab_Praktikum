import csv
import time
from pathlib import Path
import sounddevice as sd
import soundfile as sf
import numpy as np

from GenerateScript import get_script

# CONFIG
SAMPLE_RATE = 16000
CHANNELS = 1
BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "small_dataset" / "audio_test"
CSV_FILE = BASE_DIR / "small_dataset" / "transcripts_test.csv"

# MAKE SURE DIRECTORIES AND FILES EXIST
def prepare_directories():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CSV_FILE.exists():
        CSV_FILE.parent.mkdir(parents=True, exist_ok=True)

        with open(CSV_FILE, "w", newline="", encoding="ISO-8859-1") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "speaker", "text"])

# AUTOMATICALLY GENERATE FILENAME WITH UNIQUE ID
def get_next_filename():
    existing = list(OUTPUT_DIR.glob("recording_*.wav"))
    number = len(existing) + 1
    return f"recording_{number:05d}.wav"

# AUTOMATICALLY SAVE CSV DATA
def save_transcript(filename, speaker_id, transcript):
    with open(CSV_FILE, "a", newline="", encoding="ISO-8859-1") as f:
        writer = csv.writer(f)
        writer.writerow([filename, speaker_id, transcript])

# RECORD
def record_audio():
    print("Recording...")
    print("Press ENTER when finished.")
    chunks = []

    def callback(indata, frames, time_info, status):
        if status:
            print(status)

        chunks.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32",
                        callback=callback):
        input()

    if len(chunks) == 0:
        print("WARNING: No audio was recorded.")
        return None

    audio = np.concatenate(chunks, axis=0)
    return audio

# MAIN
def main():
    prepare_directories()

    print("STARTED RECORDING SESSION")
    print()
    # CHECK WHICH MICROPHONE IS BEING USED
    print(sd.query_devices())
    print()
    print("Default input/output:")
    print(sd.default.device)
    print()

    speaker_id = input("Enter speaker ID: ").strip()
    while not speaker_id:
        print("Speaker ID cannot be empty.")
        speaker_id = input("Enter speaker ID: ").strip()
    print()
    print(f"Selected: {speaker_id}")
    print()

    while True:
        phrase = get_script()
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
        sf.write(path, audio, SAMPLE_RATE, subtype="PCM_16")

        save_transcript(filename, speaker_id, phrase)
        print()
        print(f"Saved: {filename}")
        print(f"Text:  {phrase}")

if __name__ == "__main__":
    main()