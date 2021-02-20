import torch
import numpy as np
import torchaudio
from lstm_model import LSTMModel
import os
import sys
sys.path.append('../')
from datasets import LogMelSpectrogram
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils import Decoder, TextEncoder
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from beam_search import prefix_beam_search
import time
import pkg_resources
from symspellpy import SymSpell, Verbosity
from collections import defaultdict



def aggregate_predictions(beams):
    ave_word_count = 0
    for beam in beams:
        ave_word_count += len(beam.split(' '))
    ave_word_count //= len(beams)

    valid_beams = [beam.split(' ') for beam in beams if len(beam.split(' ')) == ave_word_count]
    predictions = [set() for i in range(ave_word_count)]

    for i, variations in enumerate(predictions):
        for beam in valid_beams:
            variations.add(beam[i])

    return predictions


def correct_word_spelling(variations, sym_spell):
    suggestion_counts = defaultdict(lambda: 0)

    for variant in variations:
        suggestions = sym_spell.lookup(variant, Verbosity.CLOSEST, max_edit_distance=2)

        # If any of the beams predict a real word, return that word.
        if len(suggestions) == 1:
            return suggestions[0]._term

        for suggestion in suggestions:
            suggestion_counts[suggestion._term] += 1

    ordered_suggestions = sorted(suggestion_counts, key=lambda x: suggestion_counts[x], reverse=True)
    return ordered_suggestions[0]


def correct_all_spelling(all_beam_predictions, sym_spell):
    best_predictions = []

    for word_variations in all_beam_predictions:
        best_pred = correct_word_spelling(word_variations, sym_spell)
        best_predictions.append(best_pred)
    return ' '.join(best_predictions)



if __name__ == '__main__':
    sym_spell = SymSpell(max_dictionary_edit_distance=2)
    dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

    with open('kite_reading.txt', 'r') as f:
        labels = f.readlines()

    files = [f'kite_clips_0/clip{i}.wav' for i in range(len(os.listdir('kite_clips_0')))]
    checkpoint = torch.load('ckpt-5.pth')
    model = checkpoint['model']
    model.eval()
    decoder = Decoder()

    with torch.no_grad():
        for label, file in zip(labels, files):
            waveform, sample_rate = torchaudio.load(file)
            waveform = waveform.unsqueeze(0)
            resample = torchaudio.transforms.Resample(sample_rate, 8000)
            transform = LogMelSpectrogram(8000, 400, 0.5, 128)
            waveform = resample(waveform)
            spec = transform(waveform)

            output = F.softmax(model(spec), dim=2)
            output = output.permute(1, 0, 2)
            beams = prefix_beam_search(output.squeeze(1).numpy(), k=10)

            all_beam_predictions = aggregate_predictions(beams)
            final_pred = correct_all_spelling(all_beam_predictions, sym_spell)
            print(label[:-2])
            print(final_pred)
            print()
