from other_model import SpeechRecognitionModel
from dataset import SoundClipDataset, data_processing, GreedyDecoder
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F

if __name__ == '__main__':
    dataset = SoundClipDataset(sample_rate=16000)
    loader = DataLoader(dataset, batch_size=2, collate_fn=data_processing)
    model = SpeechRecognitionModel(n_cnn_layers=3,
                                   n_rnn_layers=5,
                                   rnn_dim=512,
                                   n_class=29,
                                   n_feats=128,
                                   stride=2,
                                   dropout=0.1)
    model.load_state_dict(torch.load('trained_model/model_10.pt', map_location=torch.device('cpu')))
    model.eval()
    for (specs, labels, spec_lens, label_lens) in loader:
        print('in')
        output = model(specs)  # (batch, time, n_class)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1) # (time, batch, n_class)

        decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0, 1), labels, label_lens)
        print(decoded_preds)
        print(decoded_targets)
        input()
