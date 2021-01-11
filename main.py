from dataset import SoundClipDataset, collate_fn
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils import get_wav_length_ms, get_num_samples
import os
from model import SpeechModel, SpeechRecognition
import torch
import torch.nn as nn
from train import train_one_epoch
import torch.optim as optim
import wandb
import argparse

def num_samples(filename, data_root, win_length, overlap):
    path = os.path.join(data_root, filename)
    time = get_wav_length_ms(path)
    return get_num_samples(time, win_length, overlap)

def init_metadata(csv_path, data_root):
    df = pd.read_csv(csv_path)
    df['num_samples'] = df['path'].apply(num_samples, data_root=data_root,
                                         win_length=20, overlap=0.5)
    df = df.sort_values('num_samples', ascending=True).reset_index(drop=True)
    return df

def decode_text(preds):
    ls = []
    for i in preds[0]:
        ls.append(torch.argmax(i).item())
    return ls

def view_preds(preds):
    im = preds[0].permute(1,0).detach().numpy()
    plt.imshow(im)
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='test_clips')
    parser.add_argument('--csv_path', type=str, default='test_clips_info.csv')
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    #wandb.init()
    df = init_metadata(args.csv_path, args.data_root)
    dataset = SoundClipDataset(df, data_root=args.data_root)
    loader = DataLoader(dataset, batch_size=args.train_batch_size, collate_fn=collate_fn)
    #model = SpeechModel()
    model = SpeechRecognition()

    optimizer = optim.SGD(model.parameters(), lr=0.00001)
    ctc_loss = nn.CTCLoss(zero_infinity=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99, verbose=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i in range(10):
        train_one_epoch(model, ctc_loss, optimizer, scheduler, loader, device)

        spec, label = dataset[0]
        model.eval()
        preds = model(spec.unsqueeze(0))
        print(preds.shape)
        ls = decode_text(preds)
        #view_preds(preds)
        print('Network: ')
        print(ls)
        print('Label')
        print(label)




    '''for i in range(5):
        spec, label = dataset[0]
        plt.figure(1)
        plt.subplot(211)
        plt.imshow(spec)

        features = nn.functional.relu(model.conv1(spec.unsqueeze(0)))
        plt.subplot(212)
        plt.imshow(features.squeeze(0).detach().numpy())
        plt.show()

        for x in range(10):
            train_one_epoch(model, ctc_loss, optimizer, scheduler, loader, device)'''

    '''spec, label = dataset[0]
    plt.figure(1)
    plt.subplot(211)
    plt.imshow(spec)

    features = model.conv1(spec.unsqueeze(0))
    plt.subplot(212)
    plt.imshow(features.squeeze(0).detach().numpy())
    plt.show()'''







    '''spec, label = dataset[0]
    plt.imshow(spec)
    plt.show()




    print('setup model')

    for i in range(args.epochs):
        for x in range(10):
            train_one_epoch(model, ctc_loss, optimizer, scheduler, loader, device)



        spec, label = dataset[0]
        model.eval()
        preds = model(spec.unsqueeze(0))
        ls = decode_text(preds)
        plt.imshow(spec)
        plt.show()
        view_preds(preds)
        print('Network: ')
        print(ls)
        print('Label')
        print(label)'''







    #torch.save(model.state_dict(), 'model2.pt')














##############
