import torch
import torch.nn as nn
import torch.nn.functional as F

class SpeechModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=201, out_channels=201, kernel_size=2, stride=2)
        # input_size = num features in input
        # hidden_size = num features in the hidden state
        self.rnn = nn.RNN(input_size=201, hidden_size=512, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(in_features=512, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=28)

    def forward(self, x):
        # num_layers, batch_size, hidden_size
        h0 = torch.zeros(1, x.shape[0], 512)
        x = torch.clip(F.relu(self.conv1(x)),0, 20)
        x = x.permute(0,2,1)
        x, hidden = self.rnn(x, h0)
        x = x.permute(1,0,2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.permute(1,0,2) # batch, seq_len, proba_vals for time ste
