"""
Reference:
1. https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
2. https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua
3. Shen Pengyuan
Deep Residual Learning for Image Recognition.
@inproceedings{he2016deep,
  title={Deep residual learning for text classification},
  author={Shen Pengyuan},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
}
"""

import logging
import torch.nn.functional as F
import torch
import torch.nn as nn

__all__ = ["LSTM", "GRU"]


import torch
import torch.nn as nn


# class SimpleRNN(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes, num_layers=1):
#         super(SimpleRNN, self).__init__()
#         self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, num_classes)
#
#     def forward(self, x):
#         x = x.to(torch.float32)
#         x = x.view(len(x), 1, -1)
#         out, _ = self.rnn(x)
#         out = self.fc(out[:, -1, :])
#         return out


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTM, self).__init__()
        self.input_size = input_size  # 添加这一行
        self.lstm = nn.LSTM(self.input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.to(torch.float32)
        x = x.view(len(x), 1, -1)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.to(torch.float32)
        x = x.view(len(x), 1, -1)
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out


# def rnn_model(input_size, hidden_size, num_classes, num_layers, **kwargs):
#     model = SimpleRNN(input_size, hidden_size, num_classes, num_layers)
#     return model

def lstm(input_size, output_size, hidden_size, num_layers=1, **kwargs):
    model = LSTM(input_size, hidden_size, output_size, num_layers, **kwargs)
    return model

def gru(input_size, output_size, hidden_size, num_layers=1, **kwargs):
    model = GRU(input_size, hidden_size, output_size, num_layers, **kwargs)
    return model



