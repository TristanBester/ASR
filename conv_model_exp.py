import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import SoundClipDataset, collate_fn
from models import ConvModel
from tqdm import tqdm
from utils import Decoder

def incremental_average(ave, a, n):
    return ave + (a - ave)/n

decoder = Decoder()
dataset = SoundClipDataset(csv_path='top_thou.csv', data_root='top_thou')
data_loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

device = torch.device('cuda')
model = ConvModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5, verbose=True)
ctc_loss = nn.CTCLoss(zero_infinity=True).to(device)

for epoch in range(50):
    n = 0
    ave = 0
    pbar = tqdm(data_loader, position=0, leave=True, total=len(data_loader))
    model.train()
    for specs, labels, label_lens in pbar:
        specs = specs.to(device)
        labels = labels.to(device)
        label_lens = label_lens.to(device)

        optimizer.zero_grad()
        outputs = model(specs)
        outputs = F.log_softmax(outputs, dim=2)
        output_lens = torch.full(size=(outputs.shape[0],), fill_value=outputs.shape[1],
                                    dtype=torch.long).to(device)
        outputs = outputs.permute(1, 0, 2)

        loss = ctc_loss(outputs, labels, output_lens, label_lens)
        loss.backward()
        optimizer.step()

        #print(loss.item())
        n += 1
        ave = incremental_average(ave, loss.item(), n)
        pbar.set_description('Loss - ' + '%.4f' % loss.item())
    print()
    scheduler.step()
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

            if i == 20:
                break














#########
