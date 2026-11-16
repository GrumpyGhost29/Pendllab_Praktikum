# -*- coding: iso-8859-1 -*-
import torch
import torch.nn as nn
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence)
from vocabulary import TOKENS

class BasicCTCModel(nn.Module):
    def __init__(self, n_mels=80, hidden_size=128, num_layers=2):
        super().__init__()

        # GRU neural network to include data from previous frames in prediction
        self.gru = nn.GRU( input_size=n_mels, hidden_size=hidden_size, num_layers=num_layers,
            # Unidirectional because we need real time decoding later. True would likely produce
            # better results on prerecorded data.
            bidirectional=False,
            # Dropout to reduce overfitting with multiple layers
            dropout=0.2 if num_layers > 1 else 0.0,
        )

        # Linear classifier for token recognition
        self.classifier = nn.Linear(hidden_size, len(TOKENS),)

    def forward(self, mels, input_lengths=None, hidden=None):
        if input_lengths is not None:
            # Ignore padding in sequences
            packed = pack_padded_sequence(mels, input_lengths.cpu())

            # Run through NN
            packed_output, hidden = self.gru(packed, hidden)

            # Repad sequence
            # _ is the Python coding convention for variable that won't be used.
            x, _ = pad_packed_sequence(packed_output, total_length=mels.size(1))
        else:
            # For real time processing
            x, hidden = self.gru(mels, hidden)

        # Calculate prediction values and return log probabilities and NN hidden state
        # Log probabilities to prevent insanely small prediction values
        logits = self.classifier(x)
        log_probs = torch.log_softmax(logits, dim=-1)

        return log_probs, hidden