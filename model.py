import torch
import torch.nn as nn
import torch.nn.functional as F

class SpeechModel(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=10, stride=2)
        self.conv2 = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=5, stride=2)
        self.conv3 = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=2, stride=1)
        self.rnn = nn.RNN(input_size=channels, hidden_size=128, num_layers=5, batch_first=True)
        self.fc = nn.Linear(in_features=128, out_features=28)

    def forward(self, x):
        x = torch.clip(F.relu(self.conv1(x)),0, 20)
        x = torch.clip(F.relu(self.conv2(x)),0, 20)
        x = torch.clip(F.relu(self.conv3(x)),0, 20)
        h0 = torch.zeros(5, x.shape[0], 128)
        x = x.permute(0,2,1)
        x, hidden = self.rnn(x, h0)
        x = x.permute(1,0,2)
        x = self.fc(x).permute(1,0,2)
        return x
