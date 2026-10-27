import csv
from pathlib import Path

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from Vocabulary import token_to_id


SAMPLE_RATE = 16000
WINDOW_LENGTH = 400
HOP_LENGTH = 160
MELS = 80

mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE,
                                                     n_fft=WINDOW_LENGTH,
                                                     win_length=WINDOW_LENGTH,
                                                     hop_length=HOP_LENGTH,
                                                     n_mels=MELS)


class SpeechDataset(Dataset):

    def __init__(self, csv_path, audio_directory, speakers=None):
        self.audio_directory = Path(audio_directory)
        self.samples = []

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if speakers is not None: #If no speaker list is given load all speakers, otherwise load only listed.
                    if row["speaker"] not in speakers:
                        continue

                self.samples.append({"file": row["file"],
                                     "speaker": row["speaker"],
                                     "text": row["text"].lower().strip()})
        if speakers is not None:
            print(f"Loaded {len(self.samples)} recordings \nfrom {audio_directory} \nfor speakers {speakers}")

        else:
            print(f"Loaded {len(self.samples)} recordings \nfrom {audio_directory}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        audio_path = (self.audio_directory / sample["file"])
        waveform, sample_rate = torchaudio.load(audio_path)

        # Stereo -> mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if necessary
        if sample_rate != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)

        # Log-Mel spectrogram
        mel = mel_transform(waveform)
        mel = torch.log(mel + 1e-6)
        mel = mel.squeeze(0)

        # Per-recording normalization
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        words = sample["text"].split()

        target = torch.tensor([token_to_id[word] for word in words], mdtype=torch.long)

        return {"mel": mel, "target": target, "text": sample["text"], "speaker": sample["speaker"], "file": sample["file"]}


def collate_fn(batch):
    mels = []
    targets = []
    input_lengths = []
    target_lengths = []
    texts = []
    speakers = []
    filenames = []

    for item in batch:
        # [mel, time] -> [time, mel]
        mel = item["mel"].transpose(0, 1)
        mels.append(mel)
        targets.append(item["target"])
        input_lengths.append(
            mel.shape[0]
        )

        target_lengths.append(
            len(item["target"])
        )

        texts.append(item["text"])
        speakers.append(item["speaker"])
        filenames.append(item["file"])

    # Pad along time axis
    mels = pad_sequence(
        mels,
        batch_first=True
    )

    # CTC accepts concatenated targets
    targets = torch.cat(targets)

    return {
        "mels": mels,
        "targets": targets,

        "input_lengths": torch.tensor(
            input_lengths,
            dtype=torch.long
        ),

        "target_lengths": torch.tensor(
            target_lengths,
            dtype=torch.long
        ),

        "texts": texts,
        "speakers": speakers,
        "filenames": filenames,
    }