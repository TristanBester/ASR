import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm
from utils import Decoder
import numpy as np
from metrics import WER, CER

def validation(model, criterion, data_loader, device, logging=False):
    model = model.eval()
    decoder = Decoder()
    pbar = tqdm(data_loader, position=0, leave=True, total=len(data_loader))
    all_labels = []
    all_preds = []

    with torch.no_grad():
        counter = 0
        for spectrograms, labels, label_lens in pbar:
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)
            labels_lens = label_lens.to(device)

            preds = model(spectrograms)
            preds = F.log_softmax(preds, dim=2).permute(1,0,2)
            pred_lens = torch.full(size=(preds.shape[1],), fill_value=preds.shape[0], dtype=torch.long)
            loss = criterion(preds, labels, pred_lens, label_lens)

            greedy_preds = decoder.greedy_decode(preds.permute(1,0,2).cpu().numpy())
            all_labels += [decoder.decode_labels(labels.cpu().numpy(), pad_val=0)]
            all_preds += [decoder.collapse(greedy_preds)]

            if counter == 500:
                break
            else:
                counter += 1

            if logging:
                wandb.log({'val_loss':loss.item()})

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    total_WER = 0
    total_CER = 0

    err_bar = tqdm(zip(all_preds, all_labels), position=0, leave=True, total=len(all_labels))
    for pred, label in err_bar:
        total_WER += WER(pred, label)
        total_CER += CER(pred, label)

    ave_WER = total_WER/len(all_preds)
    ave_CER = total_CER/len(all_preds)

    if logging:
        wandb.log({'val_WER':loss.item()})
        wandb.log({'val_CER':loss.item()})


def view_progress(model, data_loader, device, file_name, sample_count=100):
    model.eval()
    decoder = Decoder()
    pbar = tqdm(data_loader, position=0, leave=True, total=len(data_loader))
    all_labels = []
    all_preds = []

    with torch.no_grad():
        counter = 0
        for spectrograms, labels, _ in pbar:
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)

            preds = model(spectrograms)
            preds = F.log_softmax(preds, dim=2).permute(1,0,2)

            greedy_preds = decoder.greedy_decode(preds.permute(1,0,2).cpu().numpy())
            all_labels += [decoder.decode_labels(labels.cpu().numpy(), pad_val=0)]
            all_preds += [decoder.collapse(greedy_preds)]

            if counter == sample_count:
                break
            else:
                counter += 1

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    with open(file_name, 'w') as f:
        for pred, label in zip(all_preds, all_labels):
            f.write(f'{label},{pred}\n')
















################
