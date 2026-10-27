import csv
from pathlib import Path

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from Vocabulary import token_to_id

# CONFIG
SAMPLE_RATE = 16000
N_FFT = 400
WINDOW_LENGTH = 400
HOP_LENGTH = 160
N_MELS = 80

# SPECTROGRAM TRANSFORMER
mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=N_FFT, win_length=WINDOW_LENGTH, hop_length=HOP_LENGTH, n_mels=N_MELS)

# DATASET CLASS
class SpeechDataset(Dataset):
    def __init__(self, csv_path, audio_directory, speakers=None):
        self.audio_directory = Path(audio_directory)
        self.samples = []

        # LOAD SAMPLES FROM CSV FILE
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # IF SPEAKER LIST IS GIVEN, SKIP ENTRIES NOT ON THE LIST
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

    # LOAD AUDIO FROM INDEX
    def __getitem__(self, index):
        sample = self.samples[index]
        audio_path = (self.audio_directory/sample["file"])
        waveform, sample_rate = torchaudio.load(audio_path)

        # CONVERT TO MONO IF NECESSARY
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # RESAMPLE IF NECESSARY
        if sample_rate != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)

        # TRANSFORM TO LOG-MEL SPECTROGRAM
        mel = mel_transform(waveform)
        mel = torch.log(mel + 1e-6)
        mel = mel.squeeze(0)

        # NORMALIZATION
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        # LOAD TARGET PHRASE
        words = sample["text"].split()
        target = torch.tensor([token_to_id[word] for word in words], dtype=torch.long)

        return {"mel": mel, "target": target, "text": sample["text"], "speaker": sample["speaker"], "file": sample["file"]}

# NORMALIZE FULL SET
def collate_fn(batch):
    # COLLECTOR LISTS FOR PADDING
    mels = []
    targets = []
    input_lengths = []
    target_lengths = []
    texts = []
    speakers = []
    filenames = []

    # COLLECT DATA
    for item in batch:
        # TRANSPOSE MEL ENTRIES FOR CONVENIENCE
        mel = item["mel"].transpose(0, 1)

        mels.append(mel)
        targets.append(item["target"])
        input_lengths.append(mel.shape[0])
        target_lengths.append(len(item["target"]))
        texts.append(item["text"])
        speakers.append(item["speaker"])
        filenames.append(item["file"])

    # PAD MELS ALONG TIME AXIS
    mels = pad_sequence(mels, batch_first=True)

    # CONCATENATE TARGETS (WE CAN DO THIS BECAUSE WE SUPPLY CTC WITH TARGET LENGTHS)
    targets = torch.cat(targets)

    # TRANSFORM RELEVANT COLLECTORS INTO TENSORS
    input_lengths = torch.tensor(input_lengths, dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)

    return {"mels": mels, "targets": targets, "input_lengths": input_lengths, "target_lengths": target_lengths, "texts": texts, "speakers": speakers, "filenames": filenames}