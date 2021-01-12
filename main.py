from dataset import SoundClipDataset, collate_fn
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
from model import SpeechModel
import torch
import torch.nn as nn
from train import train_one_epoch
import torch.optim as optim
import wandb
import argparse
from utils import Decoder
import torch.nn.functional as F
from validation import validation, view_progress

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str, default='prepared_csv/short_clips_train.csv')
    parser.add_argument('--val_csv', type=str, default='prepared_csv/short_clips_test.csv')
    parser.add_argument('--data_root', type=str, default='short_clips')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_decay', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=4)
    return parser.parse_args()

if __name__ == '__main__':
    args = init_args()
    wandb.init()

    train_dataset = SoundClipDataset(csv_path=args.train_csv, data_root=args.data_root)
    val_dataset = SoundClipDataset(csv_path=args.val_csv, data_root=args.data_root)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = SpeechModel(channels=128).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay, verbose=True)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True).to(device)

    for epoch in range(args.epochs):
        train_one_epoch(model, criterion, optimizer, train_loader, device, scheduler=scheduler, logging=True)
        validation(model, criterion, loader, device, logging=True)
        view_progress(model, test_loader, device, f'view_progress_{epoch}.txt')
        torch.save(model.state_dict(), f'ASR_model_{epoch}.pt')
