import torch
import sys
sys.path.append('../')
import torchaudio
from lstm_model import LSTMModel
import os
from datasets import LogMelSpectrogram
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils import Decoder, TextEncoder
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm


def read_files():
    with open('common_words_reading.txt', 'r') as f:
        common_labels = f.readlines()

    with open('ocean_reading.txt', 'r') as f:
        ocean_labels = f.readlines()

    with open('oranges_reading.txt', 'r') as f:
        orange_labels = f.readlines()

    with open('turtle_reading.txt', 'r') as f:
        turtle_labels = f.readlines()

    with open('kite_reading.txt', 'r') as f:
        kite_labels = f.readlines()

    with open('difficult_words.txt', 'r') as f:
        difficult_labels = f.readlines()

    common_clips_0 = [f'common_clips_0/clip{i}.wav' for i in range(len(os.listdir('common_clips_0')))]

    ocean_clips_0 = [f'ocean_clips_0/clip{i}.wav' for i in range(len(os.listdir('ocean_clips_0')))]
    ocean_clips_1 = [f'ocean_clips_1/clip{i}.wav' for i in range(len(os.listdir('ocean_clips_1')))]
    ocean_clips_2 = [f'ocean_clips_2/clip{i}.wav' for i in range(len(os.listdir('ocean_clips_2')))]

    orange_clips_0 = [f'orange_clips_0/clip{i}.wav' for i in range(len(os.listdir('orange_clips_0')))]
    orange_clips_1 = [f'orange_clips_1/clip{i}.wav' for i in range(len(os.listdir('orange_clips_1')))]

    turtle_clips_0 = [f'turtle_clips_0/clip{i}.wav' for i in range(len(os.listdir('turtle_clips_0')))]

    kite_clips_0 = [f'kite_clips_0/clip{i}.wav' for i in range(len(os.listdir('kite_clips_0')))]

    difficult_clips_0 = [f'difficult_clips_0/clip{i}.wav' for i in range(len(os.listdir('difficult_clips_0')))]

    return [
        (common_labels, common_clips_0),
        (ocean_labels, ocean_clips_0),
        (ocean_labels, ocean_clips_1),
        (ocean_labels, ocean_clips_2),
        (orange_labels, orange_clips_0),
        (orange_labels, orange_clips_1),
        (turtle_labels, turtle_clips_0),
        (difficult_labels, difficult_clips_0),
        (kite_labels, kite_clips_0)
    ]

def create_checkpoint(model, name):
    torch.save({'model':model}, name)

def load_checkpoint(path):
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    model = LSTMModel()
    model.load_state_dict(checkpoint['model'])
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CTCLoss()
    return model, optimizer, criterion

def get_spectrogram(path):
    waveform, sample_rate = torchaudio.load(path)
    waveform = waveform.unsqueeze(0)
    resample = torchaudio.transforms.Resample(sample_rate, 8000)
    transform = LogMelSpectrogram(8000, 400, 0.5, 128)
    waveform = resample(waveform)
    spec = transform(waveform)
    return spec

def strip_label(label):
    label = label.replace('\n', '')
    label = label.lower()
    label = label.lstrip()
    label = label.rstrip()
    return label

def train_one_epoch(model, optimizer, criterion, info):
    model.train()
    labels, paths = info
    encoder = TextEncoder()
    ave_loss = 0

    pbar = tqdm(enumerate(zip(paths, labels)), total=len(paths))
    for i, (path, label) in pbar:
        optimizer.zero_grad()
        spec = get_spectrogram(path)
        label = strip_label(label)
        label = encoder.char_to_int(label)
        label = torch.LongTensor(label)

        output = model(spec)
        output = F.log_softmax(output, dim=2)
        output_len = torch.full(size=(output.shape[0],),
                                 fill_value=output.shape[1],
                                 dtype=torch.long)
        output = output.permute(1, 0, 2)
        label_len = torch.LongTensor([label.shape[0]])
        loss = criterion(output, label, output_len, label_len)

        loss.backward()
        optimizer.step()
        ave_loss += loss.item()
        pbar.set_description(str(ave_loss/(i+1)))


def validate_model(model, info, file_name):
    model.eval()
    decoder = Decoder()
    labels, paths = info
    results = []

    with torch.no_grad():
        for path, label in zip(paths, labels):
            spec = get_spectrogram(path)
            label = strip_label(label)

            output = model(spec)
            output = F.log_softmax(output, dim=2)
            output = output.permute(1, 0, 2)

            decoded_output = decoder.greedy_decode(output.numpy())
            results.append((label, decoded_output))

    with open(file_name, 'w') as f:
            for label, pred in results:
                f.write(f'\n\n{label}\n{pred}')



if __name__ == '__main__':
    all_files = read_files()
    model, optimizer, criterion = load_checkpoint('checkpoint-2.pth')

    print('Training on oranges_reading.txt.')

    for i in range(10):
        for x in range(8):
            train_one_epoch(model, optimizer, criterion, all_files[x])
        print('Validating...')
        validate_model(model, all_files[0], f'common_{i}.txt')
        validate_model(model, all_files[1], f'ocean_{i}.txt')
        validate_model(model, all_files[4], f'oranges_{i}.txt')
        validate_model(model, all_files[-1], f'kite_{i}.txt')
        create_checkpoint(model, f'ckpt-{i}.pth')
