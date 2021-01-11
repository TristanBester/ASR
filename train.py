import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import wandb

def train_one_epoch(model, loss_func, optimizer, scheduler, data_loader, device):
    model = model.to(device)
    model = model.train()
    loss_func = loss_func
    scaler = GradScaler()

    counter = 0
    #pbar = tqdm(data_loader, position=0, leave=True, total=len(data_loader))
    for specs, targets, target_lengths in data_loader:
        specs = specs.to(device).float()
        targets = targets.to(device).long()
        target_lengths = torch.LongTensor(target_lengths).to(device)

        print('in: ', targets)
        print(specs.shape)

        with autocast():
            preds = model(specs).log_softmax(2).permute(1,0,2)
            input_lengths = torch.empty(preds.shape[1]).fill_(preds.shape[0])
            input_lengths = input_lengths.to(device).long()
            loss = loss_func(preds, targets.cpu(), input_lengths.cpu(), target_lengths.cpu())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        #pbar.set_description(f'Loss: {loss.item()}')
        #wandb.log({'train_loss':loss.item()})
        if counter == 0:
            break
        else:
            counter += 1
    scheduler.step()
