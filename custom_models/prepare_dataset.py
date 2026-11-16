# -*- coding: iso-8859-1 -*-
import csv
from pathlib import Path

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from custom_models.vocabulary import token_to_id

# Audio data config
SAMPLE_RATE = 16000
N_FFT = 400
WINDOW_LENGTH = 400
HOP_LENGTH = 160
N_MELS = 80

# spectrogram transformer
mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=N_FFT, win_length=WINDOW_LENGTH, hop_length=HOP_LENGTH, n_mels=N_MELS)

class SpeechDataset(Dataset):
    def __init__(self, csv_path, audio_directory, speakers=None):
        self.audio_directory = Path(audio_directory)

        # sample collector list
        self.samples = []

        # Load samples form csv
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Skip speakers not on speaker list
                if speakers is not None:
                    if row["speaker"] not in speakers:
                        continue

                self.samples.append({"file": row["file"], "speaker": row["speaker"],"text": row["text"].lower().strip()})

        if speakers is not None:
            print(f"Loaded {len(self.samples)} recordings \nfrom {audio_directory} \nfor speakers {speakers}")

        else:
            print(f"Loaded {len(self.samples)} recordings \nfrom {audio_directory}")

    def __len__(self):
        return len(self.samples)

    # load mel spectrogram from audio sample at index from samples list
    def __getitem__(self, index):
        sample = self.samples[index]
        audio_path = (self.audio_directory/sample["file"])
        waveform, sample_rate = torchaudio.load(audio_path)

        # convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # resample
        if sample_rate != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)

        # transform to log-mel spectrogram
        mel = mel_transform(waveform)
        mel = torch.log(mel + 1e-6)
        mel = mel.squeeze(0).transpose(0, 1)

        # normalize
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        # load target phrase
        words = sample["text"].split()
        target = torch.tensor([token_to_id[word] for word in words], dtype=torch.long)

        repeats = (target[1:] == target[:-1]).sum().item()
        minimum_frames = len(target) + repeats

        if mel.shape[0] < minimum_frames:
            raise ValueError(f"Audio too short for CTC target: {sample['file']}")

        return {"mel": mel, "target": target, "text": sample["text"], "speaker": sample["speaker"], "file": sample["file"]}

# Normalize and organize Batches
def collate_fn(batch):
    # Sort batch items by length
    batch = sorted(batch, key=lambda x: x["mel"].shape[0], reverse=True)

    # Output collector lists
    mels = []
    targets = []
    input_lengths = []
    target_lengths = []
    texts = []
    speakers = []
    filenames = []

    # Fill collector lists from batch items
    for item in batch:
        mel = item["mel"]
        mels.append(mel)
        input_lengths.append(mel.shape[0])

        target = item["target"]
        targets.append(target)
        target_lengths.append(len(target))

        texts.append(item["text"])

        speakers.append(item["speaker"])

        filenames.append(item["file"])

    # Pad all mels
    mels = pad_sequence(mels)

    # Concatenate all targets (we can do this because we provide target lengths)
    targets = torch.cat(targets)

    # Transform relevant outputs to tensors
    input_lengths = torch.tensor(input_lengths, dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)

    return {"mels": mels, "targets": targets, "input_lengths": input_lengths, "target_lengths": target_lengths, "texts": texts, "speakers": speakers, "filenames": filenames}