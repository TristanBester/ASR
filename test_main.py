from dataset import SoundClipDataset, collate_fn
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import argparse
from utils import Decoder
import torch.nn.functional as F
from validation import validation, view_progress
import numpy as np
from other_model import SpeechRecognitionModel
from tqdm import tqdm
from  models import SimpleModel, OtherModel, ConvModel


if __name__ == '__main__':
    dataset = SoundClipDataset()
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    decoder = Decoder()

    simple_model = SimpleModel(channels=128,
                        rnn_layers=5,
                        hidden_size=512,
                        n_classes=28)
    other_model = OtherModel(n_class=28)
    conv_model = ConvModel(128)

    for specs, labels, spec_lens, label_lens in loader:
        x_one = simple_model(specs.squeeze(1))
        x_two = other_model(specs)
        x_three = conv_model(specs)

        print(x_one.shape)
        print(x_two.shape)
        print(x_three.shape)
        break
























#################################
