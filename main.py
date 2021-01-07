from dataset import SoundClipDataset, collate_fn
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils import get_wav_length_ms, get_num_samples
import os
from model import SpeechModel
import torch
import torch.nn as nn
from train import train_one_epoch
import torch.optim as optim
import wandb

def num_samples(filename, data_root, win_length, overlap):
    path = os.path.join(data_root, filename)
    time = get_wav_length_ms(path)
    return get_num_samples(time, win_length, overlap)

def init_metadata(csv_path):
    df = pd.read_csv(csv_path)
    df['num_samples'] = df['path'].apply(num_samples, data_root='test_clips',
                                         win_length=20, overlap=0.5)
    df = df.sort_values('num_samples', ascending=True).reset_index(drop=True)
    return df

if __name__ == '__main__':
    wandb.init()
    pd.set_option('display.max_columns', None)
    df = init_metadata('test_clips_info.csv')

    dataset = SoundClipDataset(df)
    loader = DataLoader(dataset, batch_size=30, collate_fn=collate_fn)
    model = SpeechModel()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=10e-6)
    ctc_loss = nn.CTCLoss(zero_infinity=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i in range(30):
        train_one_epoch(model, ctc_loss, optimizer, scheduler, loader, device)














##############
