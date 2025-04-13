import torch
import torch.nn as nn
import string

class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=in_channels,
                                out_channels=64,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.pool_1 = nn.MaxPool2d(kernel_size=2,
                                     stride=2)
        self.conv_2 = nn.Conv2d(in_channels=64,
                                out_channels=128,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.pool_2 = nn.MaxPool2d(kernel_size=2,
                                   stride=2)
        self.conv_3 = nn.Conv2d(in_channels=128,
                                out_channels=256,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.conv_4 = nn.Conv2d(in_channels=256,
                                out_channels=256,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.pool_4 = nn.MaxPool2d(kernel_size=(2, 1))

        self.conv_5 = nn.Conv2d(in_channels=256,
                                out_channels=512,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.batch_norm_5 = nn.BatchNorm2d(512)
        self.conv_6 = nn.Conv2d(in_channels=512,
                                out_channels=512,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.batch_norm_6 = nn.BatchNorm2d(512)
        self.pool_6 = nn.MaxPool2d(kernel_size=(2, 1))
        self.conv_7 = nn.Conv2d(in_channels=512,
                                out_channels=512,
                                kernel_size=(2, 2))
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.conv_1(x)
        x = self.activation(x)
        x = self.pool_1(x)
        x = self.conv_2(x)
        x = self.activation(x)
        x = self.pool_2(x)
        x = self.conv_3(x)
        x = self.activation(x)
        x = self.conv_4(x)
        x = self.activation(x)
        x = self.pool_4(x)
        x = self.conv_5(x)
        x = self.activation(x)
        x = self.batch_norm_5(x)
        x = self.conv_6(x)
        x = self.activation(x)
        x = self.batch_norm_6(x)
        x = self.pool_6(x)
        x = self.conv_7(x)
        x = self.activation(x)
        x = x.squeeze(dim=2)
        x = x.permute(0, 2, 1)

        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.blstm_1 = nn.LSTM(input_size=512,
                               hidden_size=128,
                               dropout=0.2,
                               bidirectional=True)
        self.blstm_2 = nn.LSTM(input_size=256,
                               hidden_size=128,
                               dropout=0.2,
                               bidirectional=True)
        self.output = nn.Linear(in_features=256,
                                out_features=61)
        self.activation = nn.Softmax(dim=-1)
    
    def forward(self, x: torch.Tensor):
        x, (hn, cn) = self.blstm_1(x)
        x, (hn, cn) = self.blstm_2(x)
        x = self.output(x)
        x = self.activation(x)

        return x
class CRNN(nn.Module):
    def __init__(self, in_channels):
        super(CRNN, self).__init__()
        self.encoder = Encoder(in_channels=in_channels)
        self.decoder = Decoder()

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

sample = torch.rand(1, 1, 32, 128)
model = CRNN(in_channels=1)
print(torch.sum(model(sample), dim=-1))

