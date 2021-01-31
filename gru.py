import torch
import torch.nn as nn

if __name__ == '__main__':
    gru_bf = nn.GRU(input_size=512,
                 hidden_size=1024,
                 num_layers=1,
                 batch_first=True,
                 bidirectional=False)

    gru = nn.GRU(input_size=512,
                 hidden_size=1024,
                 num_layers=1,
                 batch_first=False,
                 bidirectional=False)

    batch = 1
    seq = 100
    input_size = 512

    in_bf = torch.randn(batch, seq, input_size)
    input = torch.randn(seq, batch, input_size)

    out_bf,_ = gru_bf(in_bf)
    out, _ = gru(input)

    print(out_bf.shape)
    print(out.shape)
