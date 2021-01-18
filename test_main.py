from dataset import SoundClipDataset, collate_fn, data_processing, GreedyDecoder
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
from model import SpeechModel
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


if __name__ == '__main__':
    dataset = SoundClipDataset()
    loader = DataLoader(dataset, batch_size=1, collate_fn=data_processing)

    model = SpeechRecognitionModel()
    ctc_loss = nn.CTCLoss(blank=28)
    optimizer = optim.AdamW(model.parameters(), lr=0.00001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     patience=10, factor=0.8,
                                                     verbose=True)

    '''scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-2,
                                            steps_per_epoch=10,
                                            epochs=10,
                                            anneal_strategy='linear')'''

    for epoch in range(500):
        for i, (specs, labels, spec_lens, label_lens) in enumerate(loader):
            optimizer.zero_grad()
            preds = model(specs)
            preds = F.log_softmax(preds, dim=2).transpose(0,1)
            loss = ctc_loss(preds, labels, spec_lens, label_lens)
            loss.backward()
            optimizer.step()
            scheduler.step(loss.item())
            print(f'Epoch: {epoch} - {loss.item()}')

            if epoch % 50 == 0:
                print(labels)
                decoded_preds, decoded_targets = GreedyDecoder(preds.transpose(0, 1), labels, label_lens)
                print(decoded_preds)
                plt.imshow(preds.permute(1,2,0).squeeze(0).detach().numpy())
                plt.colorbar()
                plt.show()
            break






















#################################
