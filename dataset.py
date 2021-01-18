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

class TextTransform:
    """Maps characters to integers and vice versa"""
    def __init__(self):
        char_map_str = """
        ' 0
        <SPACE> 1
        a 2
        b 3
        c 4
        d 5
        e 6
        f 7
        g 8
        h 9
        i 10
        j 11
        k 12
        l 13
        m 14
        n 15
        o 16
        p 17
        q 18
        r 19
        s 20
        t 21
        u 22
        v 23
        w 24
        x 25
        y 26
        z 27
        """
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        self.index_map[1] = ' '

    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = self.char_map['<SPACE>']
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return ''.join(string).replace('<SPACE>', ' ')


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
        #waveform = (waveform - waveform.mean())/waveform.std()
        #spectrogram = self.log_mel_spec_trans(waveform)

        #other:
        '''spectrogram = torchaudio.transforms.MelSpectrogram()(waveform).squeeze(0).transpose(0, 1)
        print(spectrogram.shape)




        label = self.df.iloc[idx]['cleaned_sentence']
        label = torch.LongTensor(self.encoder.char_to_int(label))
        label_len = torch.LongTensor([len(label)])'''
        #return spectrogram, label, label_len
        label = self.df.iloc[idx]['cleaned_sentence']
        return waveform, label


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


def data_processing(data, data_type="train"):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    for (waveform, label) in data:
        if data_type == 'train':
            spec =  np.log(torchaudio.transforms.MelSpectrogram()(waveform).squeeze(0).transpose(0, 1) + 1)
        elif data_type == 'valid':
            spec =  torchaudio.transforms.MelSpectrogram()(waveform).squeeze(0).transpose(0, 1)
        else:
            raise Exception('data_type should be train or valid')
        spectrograms.append(spec)
        label = torch.Tensor(TextTransform().text_to_int(label.lower()))
        labels.append(label)
        input_lengths.append(spec.shape[0]//2)
        label_lengths.append(len(label))

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, labels, input_lengths, label_lengths



def GreedyDecoder(output, labels, label_lengths, blank_label=28, collapse_repeated=True):
	arg_maxes = torch.argmax(output, dim=2)
	decodes = []
	targets = []
	for i, args in enumerate(arg_maxes):
		decode = []
		targets.append(TextTransform().int_to_text(labels[i][:label_lengths[i]].tolist()))
		for j, index in enumerate(args):
			if index != blank_label:
				if collapse_repeated and j != 0 and index == args[j -1]:
					continue
				decode.append(index.item())
		decodes.append(TextTransform().int_to_text(decode))
	return decodes, targets
