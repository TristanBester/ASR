import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import wandb

def train_one_epoch(model, loss_func, optimizer, scheduler, data_loader, device):
    model = model.to(device)
    loss_func = loss_func
    model = model.train()
    scaler = GradScaler()

    pbar = tqdm(data_loader, position=0, leave=True, total=len(data_loader))
    for specs, targets, target_lengths in pbar:
        specs = specs.to(device).float()
        targets = targets.to(device).long()
        target_lengths = torch.LongTensor(target_lengths).to(device)

        with autocast():
            preds = model(specs).log_softmax(2).permute(1,0,2)
            input_lengths = torch.empty(preds.shape[1]).fill_(preds.shape[0])
            input_lengths = input_lengths.to(device).long()
            loss = loss_func(preds, targets, input_lengths, target_lengths)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        wandb.log({'train_loss':loss.item()})
    scheduler.step()
