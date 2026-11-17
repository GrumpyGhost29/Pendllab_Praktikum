# -*- coding: iso-8859-1 -*-
import torch
import torchaudio
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT_DIR / "datasets" / "small_dataset"
AUDIO_DIR = DATASET_DIR / "audio"

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("GPU count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

    # Test a calculation on the GPU
    x = torch.randn(1000, 1000, device="cuda")
    y = x @ x
    torch.cuda.synchronize()

    print("GPU calculation successful!")
    print("Result device:", y.device)
else:
    print("CUDA is not available. Using CPU.")

waveform, sample_rate = torchaudio.load(AUDIO_DIR/"recording_00525.wav")
print("Audio loaded successfully!")