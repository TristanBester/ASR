import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, stride, dropout_p, n_freq):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, stride=stride, padding=True)
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
    def __init__(self, n_cnn_layers=6, n_rnn_layers=10, rnn_dim=512, n_class=29, n_feats=128, stride=2, dropout=0.1):
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
        start_time = time.time()
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2) # (batch, time, feature)
        x = self.fully_connected(x)
        print(f'CNN time - {time.time() - start_time}')
        start_time = time.time()
        x = self.birnn_layers(x)
        print(f'RNN time - {time.time() - start_time}')
        x = self.classifier(x)
        return x
# end of other model.

class PreActivationResidualRelu(nn.Module):
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
        x = F.relu(x)
        x = self.dropout_1(x)
        x = self.conv_1(x)
        x = self.layer_norm_2(x)
        x = F.relu(x)
        x = self.dropout_2(x)
        x = self.conv_2(x)
        x += identity
        return x

class PreActivationResidualGelu(nn.Module):
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
        x = F.gelu(x)
        x = self.dropout_1(x)
        x = self.conv_1(x)
        x = self.layer_norm_2(x)
        x = F.gelu(x)
        x = self.dropout_2(x)
        x = self.conv_2(x)
        x += identity
        return x

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout_p):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size,
                                stride, padding=1)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size,
                                stride, padding=1)
        self.dropout_1 = nn.Dropout(p=dropout_p)
        self.dropout_2 =nn.Dropout(p=dropout_p)

        if in_channels != out_channels:
            self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        identity = x
        x = F.relu(x)
        x = self.dropout_1(x)
        x = self.conv_1(x)
        x = F.relu(x)
        x = self.dropout_2(x)
        x = self.conv_2(x)

        if hasattr(self, 'projection'):
            identity = self.projection(identity)

        x += identity
        return x


class WideResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(1, 32, 3, 1)
        self.block_1 = BasicBlock(32, 64, 3, 1, 0.2)
        self.block_2 = BasicBlock(64, 64, 3, 1, 0.2)
        self.block_3 = BasicBlock(64, 128, 3, 1, 0.2)
        self.block_4 = BasicBlock(128, 128, 3, 1, 0.2)
        self.avg_pool = nn.AvgPool2d(4)

        self.fc = nn.Linear(31* 128, out_features=512)

        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=512 if i==0 else 512*2,
                             hidden_size=512, dropout=0.1, batch_first=i==0)
            for i in range(5)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(512*2, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 28)
        )

    def forward(self, x):
        x = x.permute(0,1,3,2)
        x = F.relu(self.conv_1(x))
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.avg_pool(x)
        x = x.permute(0,2,1,3).flatten(start_dim=2)
        x = self.fc(x)
        x = self.birnn_layers(x)
        x = self.classifier(x)
        return x

class OtherCNN(nn.Module):
    def __init__(self):
        super().__init__()
        n_feats = 128//2
        rnn_dim = 512
        self.cnn = nn.Conv2d(1, 32, 3, stride=2, padding=3//2)

        self.res_1 = ResidualCNN(32, 32, kernel=3, stride=1, dropout=0.1, n_feats=64)
        self.res_2 = ResidualCNN(32, 32, kernel=3, stride=1, dropout=0.1, n_feats=64)

        self.fully_connected = nn.Linear(n_feats*32, rnn_dim)
        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim if i==0 else rnn_dim*2,
                             hidden_size=rnn_dim, dropout=dropout, batch_first=i==0)
            for i in range(5)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim),  # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.res_1(x)
        x = self.res_2(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2) # (batch, time, feature)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = self.classifier(x)
        return x























###############################################################################
class ModelPreResRelu(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(1, 32, 3, stride=2, padding=3//2)
        self.res_1 = PreActivationResidualRelu(32, 3, 1, 0.1, 128//2)
        self.res_2 = PreActivationResidualRelu(32, 3, 1, 0.1, 128//2)
        self.res_3 = PreActivationResidualRelu(32, 3, 1, 0.1, 128//2)
        self.res_4 = PreActivationResidualRelu(32, 3, 1, 0.1, 128//2)
        self.res_5 = PreActivationResidualRelu(32, 3, 1, 0.1, 128//2)
        self.res_6 = PreActivationResidualRelu(32, 3, 1, 0.1, 128//2)
        self.fc_1 = nn.Linear(32 * 64, 512)

        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=512 if i==0 else 512*2,
                             hidden_size=512, dropout=0.1, batch_first=i==0)
            for i in range(5)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(512*2, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 28)
        )

    def forward(self, x):
        start_time = time.time()
        x = x.permute(0,1,3,2)
        x = F.gelu(self.conv_1(x))
        x = self.res_1(x)
        x = self.res_2(x)
        x = self.res_3(x)
        x = self.res_4(x)
        x = self.res_5(x)
        x = self.res_6(x)
        x = x.permute(0,2,1,3).flatten(start_dim=2)
        x = F.gelu(self.fc_1(x))
        x = self.birnn_layers(x)
        x = self.classifier(x)
        return x

class ModelPreResGelu(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(1, 32, 3, stride=2, padding=3//2)
        self.res_1 = PreActivationResidualGelu(32, 3, 1, 0.1, 128//2)
        self.res_2 = PreActivationResidualGelu(32, 3, 1, 0.1, 128//2)
        self.res_3 = PreActivationResidualGelu(32, 3, 1, 0.1, 128//2)
        self.res_4 = PreActivationResidualGelu(32, 3, 1, 0.1, 128//2)
        self.res_5 = PreActivationResidualGelu(32, 3, 1, 0.1, 128//2)
        self.res_6 = PreActivationResidualGelu(32, 3, 1, 0.1, 128//2)
        self.fc_1 = nn.Linear(32 * 64, 512)

        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=512 if i==0 else 512*2,
                             hidden_size=512, dropout=0.1, batch_first=i==0)
            for i in range(5)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(512*2, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 28)
        )

    def forward(self, x):
        start_time = time.time()
        x = x.permute(0,1,3,2)
        x = F.gelu(self.conv_1(x))
        x = self.res_1(x)
        x = self.res_2(x)
        x = self.res_3(x)
        x = self.res_4(x)
        x = self.res_5(x)
        x = self.res_6(x)
        x = x.permute(0,2,1,3).flatten(start_dim=2)
        x = self.fc_1(x)
        x = self.birnn_layers(x)
        x = self.classifier(x)
        return x



###############################################################################
class ModelGelu(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(1, 32, 3, stride=2, padding=3//2)
        self.res_1 = ResidualBlockGelu(32, 3, 1, 0.1, 128//2)
        self.res_2 = ResidualBlockGelu(32, 3, 1, 0.1, 128//2)
        self.res_3 = ResidualBlockGelu(32, 3, 1, 0.1, 128//2)
        self.res_4 = ResidualBlockGelu(32, 3, 1, 0.1, 128//2)
        self.res_5 = ResidualBlockGelu(32, 3, 1, 0.1, 128//2)
        self.res_6 = ResidualBlockGelu(32, 3, 1, 0.1, 128//2)
        self.fc_1 = nn.Linear(32 * 64, 512)

        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=512 if i==0 else 512*2,
                             hidden_size=512, dropout=0.1, batch_first=i==0)
            for i in range(5)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(512*2, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 28)
        )

    def forward(self, x):
        start_time = time.time()
        x = x.permute(0,1,3,2)
        x = F.gelu(self.conv_1(x))
        x = self.res_1(x)
        x = self.res_2(x)
        x = self.res_3(x)
        x = self.res_4(x)
        x = self.res_5(x)
        x = self.res_6(x)
        x = x.permute(0,2,1,3).flatten(start_dim=2)
        x = F.gelu(self.fc_1(x))
        x = self.birnn_layers(x)
        x = self.classifier(x)
        return x

class ModelRelu(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(1, 32, 3, stride=2, padding=3//2)
        self.res_1 = ResidualBlockRelu(32, 3, 1, 0.1, 128//2)
        self.res_2 = ResidualBlockRelu(32, 3, 1, 0.1, 128//2)
        self.res_3 = ResidualBlockRelu(32, 3, 1, 0.1, 128//2)
        self.res_4 = ResidualBlockRelu(32, 3, 1, 0.1, 128//2)
        self.res_5 = ResidualBlockRelu(32, 3, 1, 0.1, 128//2)
        self.res_6 = ResidualBlockRelu(32, 3, 1, 0.1, 128//2)
        self.fc_1 = nn.Linear(32 * 64, 512)

        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=512 if i==0 else 512*2,
                             hidden_size=512, dropout=0.1, batch_first=i==0)
            for i in range(5)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(512*2, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 28)
        )

    def forward(self, x):
        start_time = time.time()
        x = x.permute(0,1,3,2)
        x = F.relu(self.conv_1(x))
        x = self.res_1(x)
        x = self.res_2(x)
        x = self.res_3(x)
        x = self.res_4(x)
        x = self.res_5(x)
        x = self.res_6(x)
        x = x.permute(0,2,1,3).flatten(start_dim=2)
        x = F.relu(self.fc_1(x))
        x = self.birnn_layers(x)
        x = self.classifier(x)
        return x

class BaselineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(1, 32, 3, stride=2, padding=3//2)
        self.res_1 = ResidualBlockGelu(32, 3, 1, 0.1, 128//2)
        self.res_2 = ResidualBlockGelu(32, 3, 1, 0.1, 128//2)
        self.res_3 = ResidualBlockGelu(32, 3, 1, 0.1, 128//2)
        self.res_4 = ResidualBlockGelu(32, 3, 1, 0.1, 128//2)
        self.res_5 = ResidualBlockGelu(32, 3, 1, 0.1, 128//2)
        self.res_6 = ResidualBlockGelu(32, 3, 1, 0.1, 128//2)
        self.fc_1 = nn.Linear(32 * 64, 512)
        self.gru = nn.GRU(input_size=512, hidden_size=512, num_layers=1,
                          batch_first=True, dropout=0.1, bidirectional=True)
        self.gru2 = nn.GRU(input_size=1024, hidden_size=512, num_layers=1,
                          batch_first=False, dropout=0.1, bidirectional=True)



        self.fc_2 = nn.Linear(1024, 512)
        self.dropout = nn.Dropout(0.1)
        self.fc_3 = nn.Linear(512, 28)

    def forward(self, x):
        start_time = time.time()
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

        #print(x.shape)

        #print(f'CNN time - {start_time - time.time()}')
        start_time = time.time()

        print(x.shape)
        x, _ = self.gru(x)
        print(x.shape)
        #for i in range(9):
        #    x,_ = self.gru2(x)





        #print(f'RNN time - {start_time -  time.time()}')
        x = F.gelu(self.fc_2(x))
        x = self.dropout(x)
        x = self.fc_3(x)
        return x







































































##################################
