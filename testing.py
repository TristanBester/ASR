import torch
from models import NormConv, LSTMConv, NormLSTMConv, ResConv, ResLSTM
from dataset import SoundClipDataset, collate_fn
from torch.utils.data import DataLoader
import pandas as pd



if __name__ == '__main__':
    dataset = SoundClipDataset()
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    model = ResLSTM()

    for specs, labels, label_lens in loader:
        #print(specs.shape)
        #print(labels.shape)
        #print(label_lens)
        x = model(specs)
        print(x.shape)
        break























#########################################33
