import json
import torch
import torchaudio
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


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

class LibriSpeechDataset(Dataset):
    def __init__(self, csv_path, sample_rate=8000, window_ms=400, overlap=0.5, n_mels=128):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.log_mel_spec_trans = LogMelSpectrogram(sample_rate, window_ms,
                                                    overlap, n_mels)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        info = self.df.iloc[idx]
        waveform, _ = torchaudio.load(info['file_path'])
        spectrogram = self.log_mel_spec_trans(waveform).float()
        label = json.loads(info['int_encoded_label'])
        label = torch.LongTensor(label)
        return spectrogram, label

def collate_fn(batch, label_pad_val=0):
    spectrograms = []
    labels = []
    label_lens = []

    for spectrogram, label in batch:
        spectrograms.append(spectrogram.permute(2,0,1))
        labels.append(label)
        label_lens.append(label.shape[0])

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    spectrograms = spectrograms.permute(0, 2, 3, 1)
    label_lens = torch.LongTensor(label_lens)
    return spectrograms, labels, label_lens
