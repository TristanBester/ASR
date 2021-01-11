from test_dataset import Data,collate_fn_padd
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils import get_wav_length_ms, get_num_samples
import os
from test_model import SpeechRecognition
import torch
import torch.nn as nn
from train import train_one_epoch
import torch.optim as optim
import wandb
import argparse
from test_dataset import TextProcess
from test_train_epoch import train_epoch
import torchaudio
import numpy as np
import torch.nn.functional as F
from model import SpeechModel
import time

def GreedyDecoder(output, labels, label_lengths, blank_label=28, collapse_repeated=True):
    textprocess = TextProcess()
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    targets = []
    for i, args in enumerate(arg_maxes):
        decode = []
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j -1]:
        	           continue
                decode.append(index.item())
        decodes.append(textprocess.int_to_text_sequence(decode))

    targets = textprocess.int_to_text_sequence(labels)
    return decodes, targets


def decode(preds):
    mapping = {
        0:'-',
        1:'h',
        2:'e',
        3:'a',
        4:'b',
        5:'c',
        6:'d',
        7:'f',
        8:'g',
        9:'i',
        10:'j',
        11:'k',
        12:'l',
        13:'m',
        14:'n',
        15:'o',
        16:'p',
        17:'q',
        18:'r',
        19:'s',
        20:'t',
        21:'u',
        22:'v',
        23:'w',
        24:'x',
        25:'y',
        26:'z',
        27:'<>'
    }
    out = ''
    for ch in preds:
        out += mapping[ch]
    return out



if __name__ == '__main__':
    model = SpeechRecognition()
    my_model = SpeechModel()
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(my_model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.99, patience=1000, verbose=True)
    '''scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.5, patience=3, verbose=True)'''

    waveform, sample_rate = torchaudio.load('test_clips/common_voice_en_18637803.wav')

    he_wave = waveform[0][35000:45000].unsqueeze(0)
    ran_wave = waveform[0][45000:55000].unsqueeze(0)
    up_wave = waveform[0][55000:63000].unsqueeze(0)
    the_wave = waveform[0][65000:70000].unsqueeze(0)
    stairs_wave = waveform[0][70000:95000].unsqueeze(0)
    into_wave = waveform[0][100000:110000].unsqueeze(0)
    the_wave2 = waveform[0][110000:116000].unsqueeze(0)
    hotel_wave = waveform[0][118000:145000].unsqueeze(0)

    he_ran_wave = waveform[0][0:55000].unsqueeze(0)
    big_wave = waveform[0][0:95000].unsqueeze(0)


    resample = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=8000)
    transform = torchaudio.transforms.MelSpectrogram(sample_rate=8000,
                                                     n_mels=81,
                                                     win_length=160,
                                                     hop_length=80)



    he_wave = resample(he_wave)
    ran_wave = resample(ran_wave)
    up_wave = resample(up_wave)
    the_wave = resample(the_wave)
    stairs_wave = resample(stairs_wave)
    into_wave = resample(into_wave)
    the_wave2 = resample(the_wave2)
    hotel_wave = resample(hotel_wave)
    he_ran_wave = resample(he_ran_wave)
    big_wave = resample(big_wave)

    he_spec = np.log(transform(he_wave) + 1)
    ran_spec = np.log(transform(ran_wave) + 1)
    up_spec = np.log(transform(up_wave) + 1)
    the_spec = np.log(transform(the_wave) + 1)
    stairs_spec = np.log(transform(stairs_wave) + 1)
    into_spec = np.log(transform(into_wave) + 1)
    the_spec2 = np.log(transform(the_wave2) + 1)
    hotel_spec = np.log(transform(hotel_wave) + 1)
    he_ran_spec = np.log(transform(he_ran_wave) + 1)
    big_spec = np.log(transform(big_wave) + 1)

    he_label = torch.LongTensor([1, 2])
    ran_label = torch.LongTensor([18, 3, 14])
    up_label = torch.LongTensor([21, 16])
    the_label = torch.LongTensor([20, 1, 2])
    stairs_label = torch.LongTensor([19, 20, 3, 9, 18, 19])
    into_label = torch.LongTensor([9, 14, 20, 15])
    hotel_label = torch.LongTensor([1, 15, 20, 2, 12])
    he_ran_label = torch.LongTensor([1, 2, 27, 18, 3, 14])
    big_label = torch.LongTensor([1, 2, 27, 18, 3, 14, 27, 21, 16, 27, 20, 1, 2, 27, 19, 20, 3, 9, 18, 19])

    he_label_len = torch.LongTensor([2])
    ran_label_len = torch.LongTensor([3])
    up_label_len = torch.LongTensor([2])
    the_label_len = torch.LongTensor([3])
    stairs_label_len = torch.LongTensor([6])
    into_label_len = torch.LongTensor([4])
    hotel_label_len = torch.LongTensor([5])
    he_ran_label_len = torch.LongTensor([6])
    big_label_len = torch.LongTensor([20])

    specs = [stairs_spec, hotel_spec, he_ran_spec, big_spec]

    labels =  [stairs_label, hotel_label, he_ran_label, big_label]

    label_lens = [stairs_label_len, hotel_label_len, he_ran_label_len, big_label_len]


    my_model.train()
    for epoch in range(1000):
        for spec, label, label_len in zip(specs[:], labels[:], label_lens[:]):
            optimizer.zero_grad()
            preds = my_model(spec)
            preds = F.log_softmax(preds, dim=2).permute(1,0,2)
            spec_len = torch.LongTensor([preds.shape[0]])
            loss = ctc_loss(preds, label, spec_len, label_len)
            loss.backward()
            optimizer.step()
            scheduler.step(loss.item())
            print(loss.item())
            preds = preds.permute(1,0,2)
            preds = torch.argmax(preds, dim=2)
            preds = preds.detach().numpy().tolist()[0]
            out = decode(preds)
            print(out)


            '''optimizer.zero_grad()
            hidden = model._init_hidden(1)
            hn, c0 = hidden[0], hidden[1]
            preds,_ = model(spec, (hn, c0))
            preds = F.log_softmax(preds, dim=2)
            spec_len = torch.LongTensor([preds.shape[0]])
            loss = ctc_loss(preds, label, spec_len, label_len)
            loss.backward()
            optimizer.step()
            scheduler.step(loss.item())

            print(loss.item())
            preds = preds.permute(1,0,2)
            preds = torch.argmax(preds, dim=2)
            preds = preds.detach().numpy().tolist()[0]
            out = decode(preds)
            print(out)'''
            #break
        #break
        print()
        print()



















    '''subwave = resample(subwave)
    spec = transform(subwave)
    spec = np.log(spec + 1)

    mapping = {'h':1, 'e':2}
    label = [1,2]
    label_len = [2]

    label = torch.LongTensor(label)
    label_len = torch.LongTensor(label_len)

    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.50, patience=3)

    text_process = TextProcess()

    for epoch in range(1000):
        optimizer.zero_grad()
        hidden = model._init_hidden(1)
        hn, c0 = hidden[0], hidden[1]
        preds,_ = model(spec, (hn, c0))
        preds = F.log_softmax(preds, dim=2)
        spec_len = torch.LongTensor([preds.shape[0]])
        loss = ctc_loss(preds, label, spec_len, label_len)
        loss.backward()
        optimizer.step()
        #scheduler.step(loss.item())

        print(loss.item())
        preds = preds.permute(1,0,2)
        preds = torch.argmax(preds, dim=2)
        preds = preds.detach().numpy().tolist()[0]
        out = decode(preds)
        print(out)'''




























    '''loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn_padd)

    ctc_loss = nn.CTCLoss(blank=28, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=10e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.50, patience=6)'''


    '''for i in range(100):
        for x in range(10):
            train_epoch(model, ctc_loss, optimizer, scheduler, loader)
        model.eval()
        spec, label, spec_len, label_len = dataset[0]
        hidden = model._init_hidden(1)
        hn, c0 = hidden[0], hidden[1]
        preds, _ = model(spec, (hn, c0))
        label_len = [label_len]
        decoded, target = GreedyDecoder(preds, label, label_len)
        print(decoded)
        print(target)
    '''










#E##########
