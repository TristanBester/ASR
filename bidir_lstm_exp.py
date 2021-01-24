import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import SoundClipDataset, collate_fn
from models import BidirectionalLSTM
from tqdm import tqdm
from utils import Decoder

decoder = Decoder()
dataset = SoundClipDataset(csv_path='short_clips.csv', data_root='short_clips')
data_loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

device = torch.device('cuda')
model = BidirectionalLSTMdevice()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                                 patience=1, min_lr=10e-8, verbose=True)
ctc_loss = nn.CTCLoss(zero_infinity=True).to(device)

for epoch in range(7):
    n = 0
    epoch_loss = 0
    loader = tqdm(data_loader, position=0, leave=True, total=len(data_loader))
    model.train()

    for specs, labels, label_lens in loader:
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
        loss = ctc_loss(outputs, labels, output_lens, label_lens)
        loss.backward()
        optimizer.step()
        n += 1
        epoch_loss += loss.item()
        loader.set_description(str(loss.item()))

    print(f'\nEpoch {epoch} loss: {epoch_loss/n}')
    print('Model predictions: ')
    model.eval()

    with torch.no_grad():
        for i, (specs, labels, _) in enumerate(data_loader):
            specs = specs.to(device)
            labels = labels.to(device)
            label_lens = label_lens.to(device)

            outputs = model(specs)
            outputs = F.log_softmax(outputs, dim=2)
            preds = decoder.greedy_decode(outputs.cpu().numpy())[0]
            labels = decoder.decode_labels(labels.cpu().numpy())[0]
            print(f'{labels} - {preds}')

            if i == 10:
                break
