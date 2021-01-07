from torch.utils.data import Dataset
import torch
import torchaudio
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import string
import librosa


class TextEncoder():
    def __init__(self):
        alphabet = [chr(i) for i in range(97, 123)]
        self.mapping = {ch:i for ch,i in zip(alphabet, range(1, len(alphabet)+1))}
        self.mapping[' '] = 27
        self.translation = str.maketrans('', '', string.punctuation.replace(' ', ''))

    def encode(self, sentence):
        sentence = sentence.translate(self.translation)
        sentence = sentence.lower()
        encoded = []
        for ch in sentence:
            encoded.append(self.mapping[ch])
        return encoded


class LogSpectrogram(nn.Module):
    def __init__(self, win_length, hop_length, power=2):
        super().__init__()
        self.transform = torchaudio.transforms.Spectrogram(win_length=win_length,
                                                           hop_length=hop_length,
                                                           window_fn=torch.hann_window,
                                                           power=power)
    def forward(self, x):
        x = self.transform(x)
        return torch.log2(x + 10e-7)


class SoundClipDataset(Dataset):
    def __init__(self, df, data_root='test_clips', is_valid=False,
                 sample_rate=16000, window_ms=20, overlap=0.5):
        super().__init__()
        self.df = df
        self.data_root = data_root
        self.is_valid = is_valid
        self.sample_rate = sample_rate
        self.overlap = overlap
        self.window_time = window_ms/1000
        samples_per_window = int(self.sample_rate * self.window_time)
        hop_length = int(samples_per_window * self.overlap)
        self.log_spec_fn = LogSpectrogram(win_length=samples_per_window,
                                          hop_length=hop_length)
        self.encoder = TextEncoder()

    def __len__(self):
        return len(self.df)

    def get_spectrogram(self, waveform, sample_rate):
        resampled_waveform = torchaudio.transforms.Resample(orig_freq=sample_rate,
                                                   new_freq=self.sample_rate)(waveform)
        spectrogram = self.log_spec_fn(resampled_waveform)
        return spectrogram

    def load_audio(self, path):
        path = os.path.join(self.data_root, path)
        waveform, sample_rate = torchaudio.load(path)
        # normalize waveform.
        waveform = (waveform - waveform.mean())/waveform.std()
        spectrogram = self.get_spectrogram(waveform, sample_rate)
        return spectrogram.squeeze(dim=0)

    def __getitem__(self, idx):
        path = self.df.iloc[idx]['path']
        spectrogram = self.load_audio(path)
        sentence = self.df.iloc[idx]['sentence']
        label = self.encoder.encode(sentence)
        return spectrogram, label


def collate_fn(batch):
    target_lengths = []
    spectrograms = []
    labels = []

    for spectrogram, label in batch:
        target_lengths.append(len(label))
        spectrograms.append(spectrogram.permute(1,0))
        labels.append(torch.LongTensor(label))

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).permute(0,2,1)
    # <eos> : 27
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=27)
    return spectrograms, labels, target_lengths
