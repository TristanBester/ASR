import torch
from model import SpeechModel
from dataset import SoundClipDataset
from torch.utils.data import DataLoader
import pandas as pd

def decode_text(preds):
    ls = []
    for i in preds[0]:
        ls.append(torch.argmax(i).item())
    print(ls)




if __name__ == '__main__':
    model = SpeechModel()
    model.load_state_dict(torch.load('model.pt'))
    model.eval()
    df = pd.read_csv('test_clips_info.csv')
    dataset = SoundClipDataset(df)
    data_loader =  DataLoader(dataset, batch_size=1)

    with torch.no_grad():
        counter = 0
        for spec, label in data_loader:
            '''print(spec.shape)
            print(model(spec).shape)
            print(torch.LongTensor(label))'''
            preds = model(spec)
            print(preds)
            decode_text(preds)
            input()

            if counter == 3:
                break
            else:
                counter += 1



















#########################################33
