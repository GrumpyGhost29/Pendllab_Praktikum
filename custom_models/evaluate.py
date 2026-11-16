# -*- coding: utf-8 -*-
import csv
import time
from pathlib import Path

import jiwer
import torch
from torch.utils.data import DataLoader

from model import BasicCTCModel
from prepare_dataset import SpeechDataset, collate_fn
from vocabulary import TOKENS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT_DIR = Path(__file__).resolve().parents[1]

MODELS_DIR = ROOT_DIR / "custom_models" / "saved_models"
DATASET_DIR = ROOT_DIR / "datasets" / "small_dataset"
AUDIO_DIR = DATASET_DIR / "audio"
RESULTS_DIR = ROOT_DIR / "custom_models" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TEST_CSV = DATASET_DIR / "test.csv"
CHECKPOINT = MODELS_DIR / "best_basic_ctc_model.pt"

TRANSCRIPTS_CSV = RESULTS_DIR / "basic_transcripts.csv"
RESULTS_CSV = RESULTS_DIR / "results.csv"

MODEL_NAME = "BasicCTCModel"
SAMPLE_RATE = 16000
HOP_LENGTH = 160
BLANK_ID = 0

id_to_token = {i: token for i, token in enumerate(TOKENS)}

def decode_ctc(token_ids):
    tokens = []
    previous_id = BLANK_ID

    for current_id in token_ids:
        if current_id != previous_id and current_id != BLANK_ID:
            tokens.append(id_to_token[current_id])

        previous_id = current_id

    return tokens

def synchronize():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

dataset = SpeechDataset(TEST_CSV, AUDIO_DIR)
loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
model = BasicCTCModel().to(DEVICE)
checkpoint = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
results = []

transcripts = []
all_references = []
all_predictions = []

exact_matches = 0
total_inference_time = 0.0
total_audio_duration = 0.0

with torch.inference_mode():
    for batch in loader:
        mels = batch["mels"].to(DEVICE)
        input_lengths = batch["input_lengths"]

        reference_text = batch["texts"][0]
        filename = batch["filenames"][0]
        speaker = batch["speakers"][0]

        synchronize()
        start = time.perf_counter()

        log_probs, _ = model(mels, input_lengths)
        valid_length = int(input_lengths[0])
        predictions = (log_probs[:valid_length, 0, :].argmax(dim=-1).cpu())

        synchronize()
        elapsed = time.perf_counter() - start

        predicted_ids = predictions.tolist()
        predicted_tokens = decode_ctc(predicted_ids)
        predicted_text = " ".join(predicted_tokens)
        reference_tokens = reference_text.split()

        exact = int(reference_text.split() == predicted_tokens)
        exact_matches += exact

        metrics = jiwer.process_words(reference_text, predicted_text)

        duration = (valid_length * HOP_LENGTH / SAMPLE_RATE)
        rtf = elapsed / duration if duration > 0 else 0.0

        total_inference_time += elapsed
        total_audio_duration += duration

        all_references.append(reference_text)
        all_predictions.append(predicted_text)

        transcripts.append({
            "filename": filename,
            "speaker": speaker,
            "wer": metrics.wer,
            "exact accuracy": exact,
            "rtf": rtf,
            "deletions": metrics.deletions,
            "insertions": metrics.insertions,
            "substitutions": metrics.substitutions,
        })

        with open(TRANSCRIPTS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "filename",
                    "speaker",
                    "wer",
                    "exact accuracy",
                    "rtf",
                    "deletions",
                    "insertions",
                    "substitutions",
                ]
            )
            writer.writeheader()
            writer.writerows(transcripts)

total_wer = (jiwer.wer(all_references, all_predictions) if all_references else 0.0)
total_exact_accuracy = (exact_matches / len(dataset) if len(dataset) > 0 else 0.0)
total_rtf = (total_inference_time / total_audio_duration if total_audio_duration > 0 else 0.0)

result = {
    "model": MODEL_NAME,
    "total wer": total_wer,
    "total exact accuracy": total_exact_accuracy,
    "total rtf": total_rtf,
    "transcript file": TRANSCRIPTS_CSV.name,
}
write_header = (not RESULTS_CSV.exists() or RESULTS_CSV.stat().st_size == 0)

with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "model",
            "total wer",
            "total exact accuracy",
            "total rtf",
            "transcript file",
        ]
    )

    if write_header:
        writer.writeheader()

    writer.writerow(result)

