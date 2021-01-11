import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class SpeechModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=81, out_channels=81, kernel_size=10, stride=2)
        self.conv2 = nn.Conv1d(in_channels=81, out_channels=81, kernel_size=5, stride=2)
        self.conv3 = nn.Conv1d(in_channels=81, out_channels=81, kernel_size=2, stride=1)
        # input_size = num features in input
        # hidden_size = num features in the hidden state
        self.rnn = nn.RNN(input_size=81, hidden_size=128, num_layers=5, batch_first=True)
        self.fc1 = nn.Linear(in_features=128, out_features=28)
        self.fc2 = nn.Linear(in_features=128, out_features=28)
        self.fc3 = nn.Linear(in_features=128, out_features=28)
        self.fc = nn.Linear(in_features=512, out_features=28)

    def forward(self, x):
        # num_layers, batch_size, hidden_size
        h0 = torch.zeros(5, x.shape[0], 128)
        x = torch.clip(F.relu(self.conv1(x)),0, 20)
        #print(x.shape)
        x = torch.clip(F.relu(self.conv2(x)),0, 20)
        #print(x.shape)
        x = torch.clip(F.relu(self.conv3(x)),0, 20)
        x = x.permute(0,2,1)
        x, hidden = self.rnn(x, h0)
        x = x.permute(1,0,2)
        '''x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)'''
        x = self.fc1(x)
        #print(x.shape)
        return x.permute(1,0,2) # batch, seq_len, proba_vals for time ste




###############################
