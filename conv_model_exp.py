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


        Labels:  tensor([[19, 20,  1, 25, 27, 20, 21, 14,  5,  4]], device='cuda:0')
Outputs:  tensor([[[ -2.0903,  -1.5693,  -2.5245,  ...,  -4.3433, -19.4653, -16.1507]],

        [[ -0.3200,  -2.8227, -14.2828,  ...,  -5.8927, -26.4086, -11.1849]],

        [[ -1.5300,  -2.3656, -16.3346,  ..., -14.9188, -20.4127,  -2.7846]],

        ...,

        [[  0.0000,     -inf,     -inf,  ...,     -inf,     -inf,     -inf]],

        [[  0.0000,     -inf,     -inf,  ...,     -inf,     -inf,     -inf]],

        [[     nan,      nan,      nan,  ...,      nan,      nan,      nan]]],
       device='cuda:0', grad_fn=<PermuteBackward>)
Traceback (most recent call last):
  File "conv_model_exp.py", line 56, in <module>
    0/0
ZeroDivisionError: division by zero

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
