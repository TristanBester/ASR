import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, stride, dropout_p, n_freq):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=1)
        self.conv_2 = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, stride=stride,
                                padding=True)

    def forward(self, x):
        identity = x
        x = self.conv_1(x)
        x = F.relu(x)
        x = self.conv_2(x)
        x += identity
        x = F.relu(x)
        return x

class ResidualBlockRelu(nn.Module):
    def __init__(self, channels, kernel_size, stride, dropout_p, n_freq):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=1)
        self.conv_2 = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, stride=stride,
                                padding=True)
        self.layer_norm_1 = nn.LayerNorm(n_freq)
        self.layer_norm_2 = nn.LayerNorm(n_freq)
        self.dropout_1 = nn.Dropout(p=dropout_p)
        self.dropout_2 = nn.Dropout(p=dropout_p)

    def forward(self, x):
        identity = x
        x = self.layer_norm_1(x)
        x = self.conv_1(x)
        x = F.relu(x)
        x = self.dropout_1(x)
        x = self.layer_norm_2(x)
        x = self.conv_2(x)
        x += identity
        x = F.relu(x)
        x = self.dropout_2(x)
        return x

class ResidualBlockGelu(nn.Module):
    def __init__(self, channels, kernel_size, stride, dropout_p, n_freq):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=1)
        self.conv_2 = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, stride=stride,
                                padding=True)
        self.layer_norm_1 = nn.LayerNorm(n_freq)
        self.layer_norm_2 = nn.LayerNorm(n_freq)
        self.dropout_1 = nn.Dropout(p=dropout_p)
        self.dropout_2 = nn.Dropout(p=dropout_p)

    def forward(self, x):
        identity = x
        x = self.layer_norm_1(x)
        x = self.conv_1(x)
        x = F.gelu(x)
        x = self.dropout_1(x)
        x = self.layer_norm_2(x)
        x = self.conv_2(x)
        x += identity
        x = F.gelu(x)
        x = self.dropout_2(x)
        return x


class BaselineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(1, 32, 1, 1)
        self.res_1 = ResidualBlockGelu(32, 3, 1, 0.1, 128)
        self.res_2 = ResidualBlockGelu(32, 3, 1, 0.1, 128)
        self.res_3 = ResidualBlockGelu(32, 3, 1, 0.1, 128)
        self.res_4 = ResidualBlockGelu(32, 3, 1, 0.1, 128)
        self.res_5 = ResidualBlockGelu(32, 3, 1, 0.1, 128)
        self.res_6 = ResidualBlockGelu(32, 3, 1, 0.1, 128)
        self.fc_1 = nn.Linear(32 * 128, 512)
        self.gru = nn.GRU(input_size=512, hidden_size=512, num_layers=5,
                          batch_first=True, dropout=0.1, bidirectional=True)
        self.fc_2 = nn.Linear(1024, 512)
        self.dropout = nn.Dropout(0.1)
        self.fc_3 = nn.Linear(512, 28)

    def forward(self, x):
        x = x.permute(0,1,3,2)
        x = F.relu(self.conv_1(x))
        x = self.res_1(x)
        x = self.res_2(x)
        x = self.res_3(x)
        x = self.res_4(x)
        x = self.res_5(x)
        x = self.res_6(x)
        x = x.permute(0,2,1,3).flatten(start_dim=2)
        x = F.gelu(self.fc_1(x))
        x, _ = self.gru(x)
        x = F.gelu(self.fc_2(x))
        x = self.dropout(x)
        x = self.fc_3(x)
        return x



























##################################
