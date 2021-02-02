from datasets import LibriSpeechDataset, collate_fn
from lstm_model import LSTMModel
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from utils import Decoder
from metrics import WER, CER
import json
import argparse

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--val_batch_size', type=int, default=1)
    return parser.parse_args()

def incremental_average(ave, n_val, n):
    if ave is None:
        return n_val

    ave = ave + (n_val - ave)/float(n)
    return ave

def train_one_epoch(model, optimizer, criterion, data_loader, device,
                    scheduler):
    model.train()
    ave_loss = None
    pbar = tqdm(data_loader, leave=True, total=len(data_loader))
    for n, (specs, labels, label_lens) in enumerate(pbar):
        specs = specs.to(device)
        labels = labels.to(device)
        label_lens = label_lens.to(device)

        optimizer.zero_grad()
        outputs = model(specs)
        outputs = F.log_softmax(outputs, dim=2)
        output_lens = torch.full(size=(outputs.shape[0],),
                                 fill_value=outputs.shape[1],
                                 dtype=torch.long).to(device)
        outputs = outputs.permute(1, 0, 2)
        loss = criterion(outputs, labels, output_lens, label_lens)
        loss.backward()
        optimizer.step()

        ave_loss = incremental_average(ave_loss, loss.item(), n+1)
        pbar.set_description(f'loss - {ave_loss}')

    scheduler.step(ave_loss)
    return ave_loss

def validate(model, criterion, data_loader, device, record_file_name):
    model.eval()

    ave_CER = 0
    ave_WER = 0
    ave_loss = 0
    recorded_strings = {'predictions':[], 'labels':[]}
    decoder = Decoder()
    pbar = tqdm(data_loader, leave=True, total=len(data_loader))

    with torch.no_grad():
        for n, (specs, labels, label_lens) in enumerate(pbar):
            specs = specs.to(device)
            labels = labels.to(device)
            label_lens = label_lens.to(device)

            outputs = model(specs)
            outputs = F.log_softmax(outputs, dim=2)
            output_lens = torch.full(size=(outputs.shape[0],),
                                     fill_value=outputs.shape[1],
                                     dtype=torch.long).to(device)
            outputs = outputs.permute(1, 0, 2)
            loss = criterion(outputs, labels, output_lens, label_lens)
            ave_loss += loss.item()

            decoded_outputs = decoder.greedy_decode(outputs.cpu().numpy())
            decoded_labels = decoder.decode_label(labels.cpu().numpy()[0])

            if n < 20:
                recorded_strings['predictions'].append(decoded_outputs)
                recorded_strings['labels'].append(decoded_labels)

            for out, lbl in zip(decoded_outputs, decoded_labels):
                ave_CER += CER(out, lbl)
                ave_WER += WER(out, lbl)

    with open(record_file_name, 'w') as f:
        json.dump(recorded_strings, f)

    ave_loss /= len(data_loader)
    ave_CER /= (len(data_loader)*data_loader.batch_size)
    ave_WER /= (len(data_loader)*data_loader.batch_size)

    print(f'Average loss: {ave_loss}')
    print(f'Avereage CER: {ave_CER}')
    print(f'Avereage WER: {ave_WER}')
    return ave_loss, ave_CER, ave_WER

def checkpoint(model, optimizer, scheduler, epoch):
    ckpt = {
        'model':model.state_dict(),
        'optimizer':optimizer.state_dict(),
        'scheduler':scheduler,
        'epoch':epoch
    }
    torch.save(ckpt, f'checkpoint-{epoch}.pth')


if __name__ == '__main__':
    args = init_args()
    train_dataset = LibriSpeechDataset(csv_path='train-clean-100.csv')
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size,
                              collate_fn=collate_fn)

    val_dataset = LibriSpeechDataset(csv_path='test-clean.csv')
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size,
                            collate_fn=collate_fn)

    device = torch.device('cuda')
    model = LSTMModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     factor=0.1,
                                                     patience=1,
                                                     threshold=0.1)
    criterion = nn.CTCLoss().to(device)

    train_losses = []
    val_losese = []
    val_CERs = []
    val_WERs = []

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, optimizer, criterion, train_loader,
                                     device, scheduler)
        val_loss, val_CER, val_WER = validate(model, criterion, val_loader,
                                              device, f'preds-{epoch}.json')
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_CERs.append(val_CER)
        val_WERs.append(val_WER)
        
        checkpoint(model, optimizer, scheduler, epoch)

        np.save(f'train_losses.npy', np.array(train_losses))
        np.save(f'val_losses.npy', np.array(val_losses))
        np.save(f'val_CERs.npy', np.array(val_CERs))
        np.save(f'val_WERs.npy', np.array(val_WERs))
