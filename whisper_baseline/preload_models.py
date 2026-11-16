# -*- coding: iso-8859-1 -*-
#Run if you see any whisper checksum errors
import torch
import whisper

MODELS = ["tiny", "base", "small", "medium", "large", "turbo"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

for model_name in MODELS:
    try:
        print(f"\nLoading model: {model_name}")
        model = whisper.load_model(model_name, device=DEVICE)

    except Exception as e:
        print(f"Failed model {model_name}: {e}")