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
optimizer = optim.Adam(model.parameters(), lr=0.001)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.996, verbose=True)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                                 patience=5, min_lr=10e-10, verbose=True)
ctc_loss = nn.CTCLoss(zero_infinity=True).to(device)

for epoch in range(1000):
    print(f'Epoch: {epoch}')
    n = 0
    ave = 0
    pbar = tqdm(data_loader, position=0, leave=True, total=1000)
    model.train()
    counter = 0
    epoch_loss = 0
    for specs, labels, label_lens in pbar:
        specs = specs.to(device)
        labels = labels.to(device)
        label_lens = label_lens.to(device)

        optimizer.zero_grad()
        outputs = model(specs)
        initial_out = outputs
        outputs = F.log_softmax(outputs, dim=2)
        log_out = outputs
        output_lens = torch.full(size=(outputs.shape[0],), fill_value=outputs.shape[1],
                                    dtype=torch.long).to(device)
        outputs = outputs.permute(1, 0, 2)

        loss = ctc_loss(outputs, labels, output_lens, label_lens)
        loss.backward()
        optimizer.step()
        #print(torch.isnan(loss))
        #print(loss.item())
        epoch_loss += loss.item()


        if torch.isnan(loss):
            print('Loss is NaN: ')
            print('Labels: ', labels)
            print('i: ', initial_out)
            print('l: ', log_out)
            print('init outputs: ')
            for x in initial_out:
                print(x)
            print('log_out: ')
            for x in log_out:
                print(x)
            0/0


        '''if counter == 5:
            break
        else:
            counter += 1'''


        #n += 1
        #if not torch.isnan(loss):
        #ave = incremental_average(ave, loss.item(), n)
        pbar.set_description(f'Loss - {loss.item()}')
    #print()
    #scheduler.step()
    print(f'\nEpoch loss: {epoch_loss/1000}')
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

            if i == 5:
                break














#########
