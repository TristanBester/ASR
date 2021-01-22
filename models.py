import torch
import torch.nn as nn
import torch.nn.functional as F

# Simple model:

class SimpleModel(nn.Module):
    def __init__(self, channels, rnn_layers,
                 hidden_size, n_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=10, stride=2)
        self.conv2 = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=5, stride=2)
        self.conv3 = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=2, stride=1)
        self.rnn = nn.RNN(input_size=channels, hidden_size=hidden_size, num_layers=rnn_layers, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_size, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=n_classes)
        self.rnn_layers = rnn_layers
        self.hidden_size = hidden_size

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        h0 = torch.zeros(self.rnn_layers, x.shape[0], self.hidden_size)
        x = x.permute(0,2,1)
        x, hidden = self.rnn(x, h0)
        x = x.permute(1,0,2)
        x = F.relu(self.fc(x))
        x = self.fc2(x).permute(1,0,2)
        return x

# end of simple model.

# other model:

class CNNLayerNorm(nn.Module):
    """Layer normalization built for cnns input"""
    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous() # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous() # (batch, channel, feature, time)

class ResidualCNN(nn.Module):
    """Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf
        except with layer norm instead of batch norm
    """
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x # (batch, channel, feature, time)

class BidirectionalGRU(nn.Module):
    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x

class OtherModel(nn.Module):
    """Speech Recognition Model Inspired by DeepSpeech 2"""
    def __init__(self, n_cnn_layers=3, n_rnn_layers=5, rnn_dim=512, n_class=29, n_feats=128, stride=2, dropout=0.1):
        super(OtherModel, self).__init__()
        n_feats = n_feats//2
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3//2)  # cnn for extracting heirachal features

        # n residual cnn layers with filter size of 32
        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats)
            for _ in range(n_cnn_layers)
        ])
        self.fully_connected = nn.Linear(n_feats*32, rnn_dim)
        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim if i==0 else rnn_dim*2,
                             hidden_size=rnn_dim, dropout=dropout, batch_first=i==0)
            for i in range(n_rnn_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim),  # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2) # (batch, time, feature)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = self.classifier(x)
        return x
# end of other model.


# ConvModel

class ConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=32,
                                kernel_size=(5,3), stride=(1,1))
        self.max_pool_1 = nn.MaxPool2d(kernel_size=(2,2))
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=64,
                                kernel_size=(5,3))
        self.max_pool_2 = nn.MaxPool2d(kernel_size=(2,2))
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=128,
                                kernel_size=(3,3))
        self.conv_4 = nn.Conv2d(in_channels=128, out_channels=128,
                                kernel_size=(3,3))

        self.fc_1 = nn.Linear(in_features=128*25, out_features=256)
        self.fc_2 = nn.Linear(in_features=256, out_features=128)

        self.rnn = nn.RNN(input_size=128, hidden_size=512, num_layers=2,
                          batch_first=True, nonlinearity='tanh')

        self.classifier = nn.Linear(in_features=512, out_features=28)

    def forward(self, x):
        x = torch.clip(F.relu(self.conv_1(x)), min=0, max=20)
        x = self.max_pool_1(x)
        x = torch.clip(F.relu(self.conv_2(x)), min=0, max=20)
        x = self.max_pool_2(x)
        x = torch.clip(F.relu(self.conv_3(x)), min=0, max=20)
        x = torch.clip(F.relu(self.conv_4(x)), min=0, max=20)
        x = torch.flatten(x, start_dim=1, end_dim=2).permute(0,2,1)
        x = torch.clip(F.relu(self.fc_1(x)), min=0, max=20)
        x = torch.clip(F.relu(self.fc_2(x)), min=0, max=20)
        h0 = torch.zeros(2, x.shape[0], 512).cuda()
        x, h_l = self.rnn(x, h0)
        x = self.classifier(x)
        return x # batch, time, class
# end of conv model.


# conv model shallow

class ShallowConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=32,
                                kernel_size=(5,3), stride=(1,1))
        self.max_pool_1 = nn.MaxPool2d(kernel_size=(2,2))
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=32,
                                kernel_size=(5,3))
        self.max_pool_2 = nn.MaxPool2d(kernel_size=(2,2))
        self.conv_3 = nn.Conv2d(in_channels=32, out_channels=64,
                                kernel_size=(3,3))

        self.fc_1 = nn.Linear(in_features=1728, out_features=128)

        self.rnn = nn.RNN(input_size=128, hidden_size=128, num_layers=5,
                          batch_first=True, nonlinearity='tanh')

        self.classifier = nn.Linear(in_features=128, out_features=28)


    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = self.max_pool_1(x)
        x = F.relu(self.conv_2(x))
        x = self.max_pool_2(x)
        x = F.relu(self.conv_3(x))
        x = torch.flatten(x, start_dim=1, end_dim=2).permute(0,2,1)
        x = F.relu(self.fc_1(x))
        h0 = torch.zeros(5, x.shape[0], 128).cuda()
        x, h_l = self.rnn(x, h0)
        x = self.classifier(x)
        return x # batch, time, class


class NormConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(normalized_shape=128)
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=32,
                                kernel_size=(5,3), stride=(1,1))
        self.max_pool_1 = nn.MaxPool2d(kernel_size=(2,2))

        self.layer_norm_2 = nn.LayerNorm(normalized_shape=62)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=64,
                                kernel_size=(5,3))
        self.max_pool_2 = nn.MaxPool2d(kernel_size=(2,2))


        self.layer_norm_3 = nn.LayerNorm(normalized_shape=29)
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=128,
                                kernel_size=(3,3))


        self.layer_norm_4 = nn.LayerNorm(normalized_shape=27)
        self.conv_4 = nn.Conv2d(in_channels=128, out_channels=128,
                                kernel_size=(3,3))

        self.fc_1 = nn.Linear(in_features=128*25, out_features=256)
        self.fc_2 = nn.Linear(in_features=256, out_features=128)

        self.rnn = nn.RNN(input_size=128, hidden_size=512, num_layers=2,
                          batch_first=True, nonlinearity='tanh')

        self.classifier = nn.Linear(in_features=512, out_features=28)

    def forward(self, x):
        x = x.permute(0, 1, 3, 2)
        x = self.layer_norm_1(x)
        x = x.permute(0, 1, 3, 2)
        x = torch.clip(F.relu(self.conv_1(x)), min=0, max=20)
        x = self.max_pool_1(x)

        x = x.permute(0,1,3,2)
        x = self.layer_norm_2(x)
        x = x.permute(0,1,3,2)
        x = torch.clip(F.relu(self.conv_2(x)), min=0, max=20)
        x = self.max_pool_2(x)

        x = x.permute(0,1,3,2)
        x = self.layer_norm_3(x)
        x = x.permute(0,1,3,2)
        x = torch.clip(F.relu(self.conv_3(x)), min=0, max=20)

        x = x.permute(0,1,3,2)
        x = self.layer_norm_4(x)
        x = x.permute(0,1,3,2)
        x = torch.clip(F.relu(self.conv_4(x)), min=0, max=20)

        x = torch.flatten(x, start_dim=1, end_dim=2).permute(0,2,1)
        x = torch.clip(F.relu(self.fc_1(x)), min=0, max=20)
        x = torch.clip(F.relu(self.fc_2(x)), min=0, max=20)

        h0 = torch.zeros(2, x.shape[0], 512).cuda()
        x, h_l = self.rnn(x, h0)
        x = self.classifier(x)
        return x


class LSTMConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=32,
                                kernel_size=(5,3), stride=(1,1))
        self.max_pool_1 = nn.MaxPool2d(kernel_size=(2,2))
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=64,
                                kernel_size=(5,3))
        self.max_pool_2 = nn.MaxPool2d(kernel_size=(2,2))
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=128,
                                kernel_size=(3,3))
        self.conv_4 = nn.Conv2d(in_channels=128, out_channels=128,
                                kernel_size=(3,3))

        self.fc_1 = nn.Linear(in_features=128*25, out_features=256)
        self.fc_2 = nn.Linear(in_features=256, out_features=128)

        self.lstm = nn.LSTM(input_size=128, hidden_size=512, num_layers=2, batch_first=True)
        self.classifier = nn.Linear(in_features=512, out_features=28)

    def forward(self, x):
        x = torch.clip(F.relu(self.conv_1(x)), min=0, max=20)
        x = self.max_pool_1(x)
        x = torch.clip(F.relu(self.conv_2(x)), min=0, max=20)
        x = self.max_pool_2(x)
        x = torch.clip(F.relu(self.conv_3(x)), min=0, max=20)
        x = torch.clip(F.relu(self.conv_4(x)), min=0, max=20)
        x = torch.flatten(x, start_dim=1, end_dim=2).permute(0,2,1)
        x = torch.clip(F.relu(self.fc_1(x)), min=0, max=20)
        x = torch.clip(F.relu(self.fc_2(x)), min=0, max=20)

        h0 = torch.zeros(2, x.shape[0], 512).cuda()
        c0 = torch.zeros(2, x.shape[0], 512).cuda()
        x, _ = self.lstm(x, (h0, c0))
        x = self.classifier(x)
        return x


class NormLSTMConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(normalized_shape=128)
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=32,
                                kernel_size=(5,3), stride=(1,1))
        self.max_pool_1 = nn.MaxPool2d(kernel_size=(2,2))

        self.layer_norm_2 = nn.LayerNorm(normalized_shape=62)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=64,
                                kernel_size=(5,3))
        self.max_pool_2 = nn.MaxPool2d(kernel_size=(2,2))


        self.layer_norm_3 = nn.LayerNorm(normalized_shape=29)
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=128,
                                kernel_size=(3,3))

        self.layer_norm_4 = nn.LayerNorm(normalized_shape=27)
        self.conv_4 = nn.Conv2d(in_channels=128, out_channels=128,
                                kernel_size=(3,3))

        self.fc_1 = nn.Linear(in_features=128*25, out_features=256)
        self.fc_2 = nn.Linear(in_features=256, out_features=128)

        self.lstm = nn.LSTM(input_size=128, hidden_size=512, num_layers=2, batch_first=True)
        self.classifier = nn.Linear(in_features=512, out_features=28)

    def forward(self, x):
        x = x.permute(0, 1, 3, 2)
        x = self.layer_norm_1(x)
        x = x.permute(0, 1, 3, 2)
        x = torch.clip(F.relu(self.conv_1(x)), min=0, max=20)
        x = self.max_pool_1(x)

        x = x.permute(0,1,3,2)
        x = self.layer_norm_2(x)
        x = x.permute(0,1,3,2)
        x = torch.clip(F.relu(self.conv_2(x)), min=0, max=20)
        x = self.max_pool_2(x)

        x = x.permute(0,1,3,2)
        x = self.layer_norm_3(x)
        x = x.permute(0,1,3,2)
        x = torch.clip(F.relu(self.conv_3(x)), min=0, max=20)

        x = x.permute(0,1,3,2)
        x = self.layer_norm_4(x)
        x = x.permute(0,1,3,2)
        x = torch.clip(F.relu(self.conv_4(x)), min=0, max=20)

        x = torch.flatten(x, start_dim=1, end_dim=2).permute(0,2,1)
        x = torch.clip(F.relu(self.fc_1(x)), min=0, max=20)
        x = torch.clip(F.relu(self.fc_2(x)), min=0, max=20)

        h0 = torch.zeros(2, x.shape[0], 512).cuda()
        c0 = torch.zeros(2, x.shape[0], 512).cuda()
        x, _ = self.lstm(x, (h0, c0))
        x = self.classifier(x)
        return x


class ResConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(normalized_shape=128)
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=32,
                                kernel_size=(5,3), stride=(1,1))
        self.max_pool_1 = nn.MaxPool2d(kernel_size=(2,2))

        self.layer_norm_2 = nn.LayerNorm(normalized_shape=62)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=64,
                                kernel_size=(5,3))
        self.max_pool_2 = nn.MaxPool2d(kernel_size=(2,2))

        self.layer_norm_3 = nn.LayerNorm(normalized_shape=29)
        self.rcv_1 = nn.Conv2d(in_channels=64,out_channels=64, kernel_size=1)
        self.rcv_2 = nn.Conv2d(in_channels=64,out_channels=64, kernel_size=3,
                              stride=1, padding=1)

        self.layer_norm_4 = nn.LayerNorm(normalized_shape=29)
        self.rcv_3 = nn.Conv2d(in_channels=64,out_channels=64, kernel_size=1)
        self.rcv_4 = nn.Conv2d(in_channels=64,out_channels=64, kernel_size=3,
                               stride=1, padding=1)

        self.fc_1 = nn.Linear(in_features=64*29, out_features=256)
        self.fc_2 = nn.Linear(in_features=256, out_features=128)

        self.rnn = nn.RNN(input_size=128, hidden_size=512, num_layers=2,
                          batch_first=True, nonlinearity='tanh')

        self.classifier = nn.Linear(in_features=512, out_features=28)

    def forward(self, x):
        x = torch.clip(F.relu(self.conv_1(x)), min=0, max=20)
        x = self.max_pool_1(x)
        x = torch.clip(F.relu(self.conv_2(x)), min=0, max=20)
        x = self.max_pool_2(x)

        residual = x
        x = x.permute(0,1,3,2)
        x = self.layer_norm_3(x)
        x = x.permute(0,1,3,2)
        x = F.relu(self.rcv_1(x))
        x = F.relu(self.rcv_2(x))
        x += residual

        residual = x
        x = x.permute(0,1,3,2)
        self.layer_norm_4(x)
        x = x.permute(0,1,3,2)
        x = F.relu(self.rcv_3(x))
        x = F.relu(self.rcv_4(x))
        x += residual

        x = torch.flatten(x, start_dim=1, end_dim=2).permute(0,2,1)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))

        h0 = torch.zeros(2, x.shape[0], 512).cuda()
        x, h_l = self.rnn(x, h0)
        x = self.classifier(x)
        return x


class ResLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(normalized_shape=128)
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=32,
                                kernel_size=(5,3), stride=(1,1))
        self.max_pool_1 = nn.MaxPool2d(kernel_size=(2,2))

        self.layer_norm_2 = nn.LayerNorm(normalized_shape=62)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=64,
                                kernel_size=(5,3))
        self.max_pool_2 = nn.MaxPool2d(kernel_size=(2,2))

        self.layer_norm_3 = nn.LayerNorm(normalized_shape=29)
        self.rcv_1 = nn.Conv2d(in_channels=64,out_channels=64, kernel_size=1)
        self.rcv_2 = nn.Conv2d(in_channels=64,out_channels=64, kernel_size=3,
                              stride=1, padding=1)

        self.layer_norm_4 = nn.LayerNorm(normalized_shape=29)
        self.rcv_3 = nn.Conv2d(in_channels=64,out_channels=64, kernel_size=1)
        self.rcv_4 = nn.Conv2d(in_channels=64,out_channels=64, kernel_size=3,
                               stride=1, padding=1)

        self.fc_1 = nn.Linear(in_features=64*29, out_features=256)
        self.fc_2 = nn.Linear(in_features=256, out_features=128)

        self.lstm = nn.LSTM(input_size=128, hidden_size=512, num_layers=2, batch_first=True)
        self.classifier = nn.Linear(in_features=512, out_features=28)

    def forward(self, x):
        x = torch.clip(F.relu(self.conv_1(x)), min=0, max=20)
        x = self.max_pool_1(x)
        x = torch.clip(F.relu(self.conv_2(x)), min=0, max=20)
        x = self.max_pool_2(x)

        residual = x
        x = x.permute(0,1,3,2)
        x = self.layer_norm_3(x)
        x = x.permute(0,1,3,2)
        x = F.relu(self.rcv_1(x))
        x = F.relu(self.rcv_2(x))
        x += residual

        residual = x
        x = x.permute(0,1,3,2)
        self.layer_norm_4(x)
        x = x.permute(0,1,3,2)
        x = F.relu(self.rcv_3(x))
        x = F.relu(self.rcv_4(x))
        x += residual

        x = torch.flatten(x, start_dim=1, end_dim=2).permute(0,2,1)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))

        h0 = torch.zeros(2, x.shape[0], 512).cuda()
        c0 = torch.zeros(2, x.shape[0], 512).cuda()
        x, _ = self.lstm(x, (h0, c0))
        x = self.classifier(x)
        return x








#end
