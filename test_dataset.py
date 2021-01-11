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

class TextProcess:
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
        self.translation = str.maketrans('', '', string.punctuation.replace(' ', ''))

        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        self.index_map[1] = ' '

    def text_to_int_sequence(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        text = text.lower()
        text = text.translate(self.translation)
        for c in text:
            if c == ' ':
            	ch = self.char_map['<SPACE>']
            else:
            	ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text_sequence(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            if i != 28:
                string.append(self.index_map[i])
            else:
                string.append('-')
        return ''.join(string).replace('<SPACE>', '')


class LogMelSpec(nn.Module):

    def __init__(self, sample_rate=8000, n_mels=128, win_length=160, hop_length=80):
        super(LogMelSpec, self).__init__()
        self.transform = torchaudio.transforms.MelSpectrogram(
                            sample_rate=sample_rate, n_mels=n_mels,
                            win_length=win_length, hop_length=hop_length)

    def forward(self, x):
        x = self.transform(x)  # mel spectrogram
        x = np.log(x + 1e-14)  # logrithmic, add small value to avoid inf
        return x


class Data(torch.utils.data.Dataset):

    # this makes it easier to be ovveride in argparse
    parameters = {
        "sample_rate": 8000, "n_feats": 81,
        "specaug_rate": 0.5, "specaug_policy": 3,
        "time_mask": 70, "freq_mask": 15
    }

    def __init__(self, csv_path='test_clips_info.csv', sample_rate=8000, n_feats=81, specaug_rate=0.5, specaug_policy=3,
                time_mask=70, freq_mask=15, valid=False, shuffle=True, text_to_int=True, log_ex=True):
        self.log_ex = log_ex
        self.text_process = TextProcess()

        #print("Loading data csv file from", csv_path)
        self.data = pd.read_csv(csv_path)
        #print("Loading successful.")

        if valid:
            self.audio_transforms = torch.nn.Sequential(
                LogMelSpec(sample_rate=sample_rate, n_mels=n_feats,  win_length=160, hop_length=80)
            )
        else:
            self.audio_transforms = torch.nn.Sequential(
                LogMelSpec(sample_rate=sample_rate, n_mels=n_feats,  win_length=160, hop_length=80),
                #SpecAugment(specaug_rate, specaug_policy, freq_mask, time_mask)
            )


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        #try:
        file_path = self.data.iloc[idx]['path']
        file_path = os.path.join('test_clips', file_path)
        print(file_path)
        waveform, sample_rate  = torchaudio.load(file_path)
        print(sample_rate)

        print(waveform.shape)
        #plt.plot(waveform.t().numpy())
        #plt.show()
        sub_wav2 = waveform[0][35000:45000]
        #plt.plot(sub_wav.t().numpy())
        #plt.show()
        #torchaudio.save('wave.wav', waveform, sample_rate=sample_rate)
        torchaudio.save('sound2.wav', sub_wav2, sample_rate=sample_rate)

        '''label = self.text_process.text_to_int_sequence(self.data.iloc[idx]['sentence'])
        spectrogram = self.audio_transforms(waveform) # (channel, feature, time)
        spec_len = spectrogram.shape[-1] // 2
        label_len = len(label)'''

        '''
        spectrogram = self.audio_transforms(waveform) # (channel, feature, time)
        spec_len = spectrogram.shape[-1] // 2
        label_len = len(label)
        if spec_len < label_len:
            raise Exception('spectrogram len is bigger then label len')
        if spectrogram.shape[0] > 1:
            raise Exception('dual channel, skipping audio file %s'%file_path)
        if spectrogram.shape[2] > 1650:
            raise Exception('spectrogram to big. size %s'%spectrogram.shape[2])
        if label_len == 0:
            raise Exception('label len is zero... skipping %s'%file_path)'''
        #except Exception as e:
        '''if self.log_ex:
            print(str(e), file_path)
        return self.__getitem__(idx - 1 if idx != 0 else idx + 1)'''
        return spectrogram, label, spec_len, label_len

    def describe(self):
        return self.data.describe()



def collate_fn_padd(data):
    '''
    Padds batch of variable length
    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    # print(data)
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    for (spectrogram, label, input_length, label_length) in data:
        if spectrogram is None:
            continue
       # print(spectrogram.shape)
        spectrograms.append(spectrogram.squeeze(0).transpose(0, 1))
        labels.append(torch.Tensor(label))
        input_lengths.append(input_length)
        label_lengths.append(label_length)

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    input_lengths = input_lengths
    # print(spectrograms.shape)
    label_lengths = label_lengths
    # ## compute mask
    # mask = (batch != 0).cuda(gpu)
    # return batch, lengths, mask
    return spectrograms, labels, input_lengths, label_lengths
