from torch.utils.data import Dataset
import torch
import torchaudio
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import string
import librosa
import pandas as pd
from utils import TextEncoder

class LogMelSpectrogram(nn.Module):
    def __init__(self, sample_rate, window_ms, overlap, n_mels):
        super().__init__()
        self.transform = torchaudio.transforms.MelSpectrogram(sample_rate = sample_rate,
                                                   win_length = window_ms,
                                                   hop_length = int(window_ms * overlap),
                                                   n_mels = n_mels)
    def forward(self, x):
        x = self.transform(x)
        return torch.log1p(x)

class SoundClipDataset(Dataset):
    def __init__(self, csv_path='prepared_csv/short_clips.csv', data_root='short_clips',
                 is_valid=False, sample_rate=8000, window_ms=400, overlap=0.5, n_mels=128):
        self.df = pd.read_csv(csv_path)
        self.data_root = data_root
        self.is_valid = is_valid
        self.sample_rate = sample_rate
        self.log_mel_spec_trans = LogMelSpectrogram(sample_rate, window_ms,
                                                    overlap, n_mels)
        self.encoder = TextEncoder()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df.iloc[idx]['path']
        path = os.path.join(self.data_root, file_name)
        waveform, orig_sample_rate = torchaudio.load(path)
        waveform = torchaudio.transforms.Resample(orig_freq=orig_sample_rate,
                                              new_freq=self.sample_rate)(waveform)
        waveform = (waveform - waveform.mean())/waveform.std()
        spectrogram = self.log_mel_spec_trans(waveform)
        label = self.df.iloc[idx]['cleaned_sentence']
        label = torch.LongTensor(self.encoder.char_to_int(label))
        label_len = torch.LongTensor([len(label)])
        return spectrogram, label, label_len


def collate_fn(batch, label_pad_val=0):
    spectrograms = []
    labels = []
    label_lens = []

    for spectrogram, label, label_len  in batch:
        spectrograms.append(spectrogram.squeeze(0).permute(1,0))
        labels.append(label)
        label_lens.append(label_len)

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).permute(0,2,1)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=label_pad_val)
    label_lens = torch.LongTensor(label_lens)
    return spectrograms, labels, label_lens
