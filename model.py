import torch
import torch.nn as nn
from typing import List
from utils import decode_targets

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
                               batch_first=True,
                               bidirectional=True,
                               num_layers=2)
        self.blstm_2 = nn.LSTM(input_size=256,
                               hidden_size=128,
                               dropout=0.2,
                               batch_first=True,
                               bidirectional=True,
                               num_layers=2)
        self.output = nn.Linear(in_features=256,
                                out_features=63)
    
    def forward(self, x: torch.Tensor):
        x, (hn, cn) = self.blstm_1(x)
        x, (hn, cn) = self.blstm_2(x)
        x = self.output(x)
        x = x.permute(1, 0, 2)
        return x
    
class CRNN(nn.Module):
    def __init__(self, in_channels):
        super(CRNN, self).__init__()
        self.encoder = Encoder(in_channels=in_channels)
        self.decoder = Decoder()

    def forward(self, x: torch.Tensor, targets: List[torch.Tensor] = None):
        batch_size, c, h, w = x.size()
        x = self.encoder(x)
        x = self.decoder(x)

        log_probs = torch.nn.functional.log_softmax(x, dim=2)

        if targets is not None:
            input_lengths = torch.full(
                size=(batch_size, ),
                fill_value=x.size(0),
                dtype=torch.int32
            )

            target_lengths = torch.zeros((batch_size, ), 
                                         dtype=torch.int32)
            for i, target in enumerate(targets):
                target_lengths[i] = len(target)

            targets = torch.cat(tensors=targets) 

            loss = nn.CTCLoss(blank=0)(
                log_probs, 
                targets,
                input_lengths,
                target_lengths
            )

            return log_probs.permute(1, 0, 2), loss
        return log_probs.permute(1, 0, 2)

sample = torch.ones((1, 1, 32, 128))
a = torch.randint(low=1, high=61, size=(12,), dtype=torch.long)
b = torch.randint(low=1, high=61, size=(21,), dtype=torch.long)

targets = [a, b]
model = CRNN(in_channels=1)
output = model(sample)
print(decode_targets(output))