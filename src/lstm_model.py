import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalized_shape):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=normalized_shape)
        self.network_in_network = nn.Conv2d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=1)
        self.conv = nn.Conv2d(in_channels=out_channels,
                              out_channels=out_channels,
                              kernel_size=3,
                              padding=1)

    def forward(self, x):
        identity = x
        x = x.permute(0,1,3,2)
        x = self.layer_norm(x)
        x = x.permute(0,1,3,2)
        x = F.gelu(self.network_in_network(x))
        x = F.gelu(self.conv(x))
        x += identity
        return x


class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5,3))
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5,3))

        self.res_block_1 = ResidualBlock(in_channels=64, out_channels=64,
                                         normalized_shape=120)
        self.res_block_2 = ResidualBlock(in_channels=64, out_channels=64,
                                         normalized_shape=120)

        self.fc_1 = nn.Linear(in_features=64*120, out_features=256)
        self.fc_2 = nn.Linear(in_features=256, out_features=128)

        self.lstm = nn.LSTM(input_size=128, hidden_size=512, num_layers=2,
                            batch_first=True)
        self.classifier = nn.Linear(in_features=512, out_features=28)

    def forward(self, x):
        x = F.gelu(self.conv_1(x))
        x = F.gelu(self.conv_2(x))

        x = self.res_block_1(x)
        x = self.res_block_2(x)

        x = torch.flatten(x, start_dim=1, end_dim=2).permute(0,2,1)
        x = F.gelu(self.fc_1(x))
        x = F.gelu(self.fc_2(x))

        x, _ = self.lstm(x)
        x = self.classifier(x)
        return x
