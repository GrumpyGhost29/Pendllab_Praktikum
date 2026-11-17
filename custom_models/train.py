# -*- coding: iso-8859-1 -*-
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from model import BasicCTCModel
from prepare_dataset import SpeechDataset, collate_fn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 1e-3
PATIENCE = 15

ROOT_DIR = Path(__file__).resolve().parents[1]

MODELS_DIR = ROOT_DIR / "custom_models" / "saved_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATASET_DIR = ROOT_DIR / "datasets" / "small_dataset"
AUDIO_DIR = DATASET_DIR/"audio"

TRAIN_CSV = DATASET_DIR/"train.csv"
VAL_CSV = DATASET_DIR/"validation.csv"
CHECKPOINT = MODELS_DIR/"best_basic_ctc_model.pt"

train_dataset = SpeechDataset(TRAIN_CSV, AUDIO_DIR)
val_dataset = SpeechDataset(VAL_CSV, AUDIO_DIR)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

model = BasicCTCModel().to(DEVICE)

criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

def run_epoch(loader, training):
    model.train(training)
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        mels = batch["mels"].to(DEVICE)
        targets = batch["targets"].to(DEVICE)
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            log_probs, _ = model(mels, input_lengths)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        batch_size = mels.size(1)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples

best_val_loss = float("inf")
epochs_without_improvement = 0

for epoch in range(1, EPOCHS + 1):
    print(f"Running next epoch")
    train_loss = run_epoch(train_loader, training=True)
    val_loss = run_epoch(val_loader, training=False)
    print(f"\n\n\nEpoch {epoch:03d} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, CHECKPOINT)
        print("---Saved new best model---")

    else:
        epochs_without_improvement += 1
        print("---No improvement so far---")

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping")
            break