import torch
#from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import wandb
import torch.nn.functional as F

def train_one_epoch(model, criterion, optimizer, data_loader, device, scheduler=None, logging=False):
    model = model.train()
    pbar = tqdm(data_loader, position=0, leave=True, total=len(data_loader))
    pbar.set_description('Training: ')

    for spectrograms, labels, label_lens in pbar:
        spectrograms = spectrograms.to(device)
        labels = labels.to(device)
        labels_lens = label_lens.to(device)
        optimizer.zero_grad()

        preds = model(spectrograms)
        preds = F.log_softmax(preds, dim=2).permute(1,0,2)
        pred_lens = torch.full(size=(preds.shape[1],), fill_value=preds.shape[0], dtype=torch.long).to(device)
        loss = criterion(preds, labels, pred_lens, label_lens)

        loss.backward()
        optimizer.step()

        if logging:
            wandb.log({'train_loss':loss.item()})

    if scheduler is not None:
        scheduler.step()
