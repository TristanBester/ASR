import torch
import torch.nn.functional as F
from tqdm import tqdm

def train_epoch(model, criterion, optimizer, scheduler, loader):
    model.train()

    pbar = tqdm(loader)
    for specs, labels, spec_lens, target_lengths in pbar:
        spec_lens = torch.LongTensor(spec_lens)
        target_lengths = torch.LongTensor(target_lengths)
        optimizer.zero_grad()
        hidden = model._init_hidden(loader.batch_size)
        hn, c0 = hidden[0], hidden[1]
        output, _ = model(specs, (hn, c0))
        output = F.log_softmax(output, dim=2)
        loss = criterion(output, labels, spec_lens, target_lengths)
        loss.backward()
        optimizer.step()
        pbar.set_description(f'Loss item: {loss.item()}')
        scheduler.step(loss.item())
        break
