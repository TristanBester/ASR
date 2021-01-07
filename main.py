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
import argparse

def num_samples(filename, data_root, win_length, overlap):
    path = os.path.join(data_root, filename)
    time = get_wav_length_ms(path)
    return get_num_samples(time, win_length, overlap)

def init_metadata(csv_path, data_root):
    df = pd.read_csv(csv_path)
    df['num_samples'] = df['path'].apply(num_samples, data_root=data_root,
                                         win_length=20, overlap=0.5)
    df = df.sort_values('num_samples', ascending=True).reset_index(drop=True)
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='test_clips')
    parser.add_argument('--csv_path', type=str, default='test_clips_info.csv')
    parser.add_argument('--train_batch_size', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()


    wandb.init()
    pd.set_option('display.max_columns', None)

    df = init_metadata(args.csv_path, args.data_root)
    print('init metadata')
    dataset = SoundClipDataset(df, data_root=args.data_root)
    loader = DataLoader(dataset, batch_size=args.train_batch_size, collate_fn=collate_fn)
    model = SpeechModel()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=10e-6)
    ctc_loss = nn.CTCLoss(zero_infinity=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('setup model')

    for i in range(args.epochs):
        print('training one epoch')
        train_one_epoch(model, ctc_loss, optimizer, scheduler, loader, device)
    torch.save(model.state_dict(), 'model.pt')














##############
