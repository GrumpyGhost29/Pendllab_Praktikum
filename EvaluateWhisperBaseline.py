# -*- coding: iso-8859-1 -*-

import gc
import time
import whisper
import pandas as pd
import torch

from pathlib import Path
from jiwer import wer, cer

from Levenshtein_Preprocessing import normalize_text

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "small_dataset"

INPUT_FILE = DATA_DIR / "test.csv"
AUDIO_DIR = DATA_DIR / "audio"

OUTPUT_FILE = DATA_DIR / "performance.csv"
OUTPUT_DIR = DATA_DIR / "results"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = ["tiny", "base", "small", "medium", "large", "turbo"]
MODEL_COMPARISON_COLUMNS = []

MODEL_RESULT_COLUMNS = ["audiofile", "WER", "CER", "RTF", "duration", "elapsed", "target", "result"]
MODEL_COMPARISON_COLUMNS = ["model", "WER", "CER", "RTF"]


def prepare_files():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not OUTPUT_FILE.exists():
        pd.DataFrame(columns=MODEL_COMPARISON_COLUMNS).to_csv(OUTPUT_FILE, index=False, encoding="ISO-8859-1")

def transcribe_audio(model, audio):
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    #record transcription time for RTF
    start = time.perf_counter()
    result = model.transcribe(audio, task="transcribe", language="de", verbose=False, fp16=(DEVICE == "cuda"), temperature=0.0)

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    return result["text"], elapsed


def evaluate_model(model_name, test_df):
    result_file = OUTPUT_DIR / f"{model_name}.csv"

    # Load previous results to support resuming
    if result_file.exists():
        results_df = pd.read_csv(result_file, encoding="ISO-8859-1", keep_default_na=False)
        if not set(MODEL_RESULT_COLUMNS).issubset(results_df.columns):
            raise ValueError(f"Invalid CSV columns in {result_file}")
    else:
        results_df = pd.DataFrame(columns=MODEL_RESULT_COLUMNS)

    completed = set(results_df["audiofile"].astype(str))
    expected = set(test_df["file"].astype(str))

    if expected.issubset(completed):
        print(f"Skipping {model_name}: already evaluated.")
        return results_df

    print(f"\nLoading model: {model_name}")

    model = whisper.load_model(model_name, device=DEVICE)

    try:
        if DEVICE == "cuda":
            print("Warming up GPU...")
            dummy_audio = torch.zeros(16000, dtype=torch.float32).numpy()
            model.transcribe(dummy_audio, language="de", fp16=True, verbose=False)
            torch.cuda.synchronize()

        for index, row in test_df.iterrows():
            audio_name = str(row["file"])
            target = str(row["text"])

            if audio_name in completed:
                continue

            audio_path = AUDIO_DIR / audio_name

            if not audio_path.is_file():
                print(f"Missing audio file: {audio_path}")
                continue

            print(f"[{model_name}] {index + 1}/{len(test_df)}: {audio_name}")

            try:
                audio = whisper.load_audio(str(audio_path))
                duration = (len(audio) / whisper.audio.SAMPLE_RATE)

                if duration <= 0:
                    print(f"Empty audio: {audio_name}")
                    continue

                prediction, elapsed = transcribe_audio(model, audio)
                prediction = normalize_text(prediction)

                word_error = wer(target, prediction)
                char_error = cer(target, prediction)
                rtf = elapsed / duration

                new_row = {"audiofile": audio_name, "WER": word_error, "CER": char_error, "RTF": rtf, "duration": duration, "elapsed": elapsed,  "target": target, "result": prediction}
                pd.DataFrame([new_row], columns=MODEL_RESULT_COLUMNS).to_csv(result_file, mode="a", header=not result_file.exists(), index=False, encoding="ISO-8859-1")
                completed.add(audio_name)

                print(f"WER: {word_error:.4f} | CER: {char_error:.4f} | RTF: {rtf:.4f}")

            except Exception as e:
                print(f"Error processing {audio_name}: {e}")

    finally:
        del model
        gc.collect()

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    if result_file.exists():
        return pd.read_csv(result_file, encoding="ISO-8859-1", keep_default_na=False)

    return pd.DataFrame(columns=MODEL_RESULT_COLUMNS)


def save_overall_results(model_name, results_df, test_df):
    if results_df.empty:
        print(f"No results for {model_name}")
        return

    expected = set(test_df["file"].astype(str))
    completed = set(results_df["audiofile"].astype(str))

    missing = expected - completed

    if missing:
        print(
            f"Skipping overall results for {model_name}: "
            f"{len(missing)} recordings missing."
        )
        return

    targets = results_df["target"].fillna("").astype(str).tolist()
    predictions = results_df["result"].fillna("").astype(str).tolist()

    overall_wer = wer(targets, predictions)
    overall_cer = cer(targets, predictions)
    overall_rtf = (results_df["elapsed"].sum()/results_df["duration"].sum())

    overall_df = pd.read_csv(OUTPUT_FILE, encoding="ISO-8859-1")

    # Remove existing entry to avoid duplicates
    overall_df = overall_df[overall_df["model"] != model_name]

    new_row = pd.DataFrame([{"model": model_name, "WER": overall_wer, "CER": overall_cer, "RTF": overall_rtf}])

    overall_df = pd.concat([overall_df, new_row], ignore_index=True)
    overall_df = overall_df.sort_values(by="WER", ascending=True)

    overall_df.to_csv(OUTPUT_FILE, index=False, encoding="ISO-8859-1")

    print(f"\n{model_name} averages:\nWER: {overall_wer:.4f}\nCER: {overall_cer:.4f}\nRTF: {overall_rtf:.4f}")


def main():
    print(f"Using device: {DEVICE}")

    prepare_files()

    test_df = pd.read_csv(INPUT_FILE, encoding="ISO-8859-1", keep_default_na=False)

    required_columns = {"file", "speaker", "text"}

    if not required_columns.issubset(test_df.columns):
        raise ValueError(f"Input CSV must contain: {required_columns}")

    if test_df["file"].duplicated().any():
        raise ValueError("Duplicate audio filenames found in Input CSV")

    print(f"Test samples: {len(test_df)}")
    print(f"Models: {MODELS}")

    for model_name in MODELS:
        try:
            results_df = evaluate_model(model_name, test_df)
            save_overall_results(model_name, results_df, test_df)

        except Exception as e:
            print(f"Failed model {model_name}: {e}")

    print("\nEvaluation complete!")
    print(f"Overall results: {OUTPUT_FILE}")
    print(f"Individual results: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
