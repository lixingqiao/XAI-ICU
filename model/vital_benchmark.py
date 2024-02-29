'''
Neural Network models, implemented by PyTorch
https://github.com/zhangxu0307/time_series_forecasting_pytorch/blob/master/src/model.py
'''
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import sys


root_dir = './../'
sys.path.append(root_dir)
from .layer_norm import LayerNorm


# 标准RNN模型
class VitalRNN(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cell = nn.RNN(input_size=config["input_size"], hidden_size=config["hidden_size"],
                           num_layers=config['n_blocks'], dropout=config["drop_out"],
                           nonlinearity=config['nonlinearity'], batch_first=True)
        self.fc = nn.Linear(in_features=config["hidden_size"], out_features=config["n_classes"], bias=True)

    def forward(self, x):
        # input shape: (batch_size, seq_len, dim)
        # batch_size = x.size(0)
        # h0 = Variable(torch.zeros(self.config['n_blocks'] * 1, batch_size, self.config["hidden_size"]))
        # h0 = h0.cuda()

        # out, _ = self.cell(x, h0)
        out, _ = self.cell(x)
        # out shape: (batch_size, seq_len, hidden_size)
        out = out[:, -1, :]
        # out shape: （bach_size, hidden_size）
        logits = self.fc(out)

        return logits


# LSTM
class VitalLSTM(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cell = nn.LSTM(input_size=config["input_size"], hidden_size=config["hidden_size"],
                            num_layers=config['n_blocks'], dropout=config["drop_out"],
                            batch_first=True)
        self.fc = nn.Linear(in_features=config["hidden_size"], out_features=config["n_classes"], bias=True)

    def forward(self, x):
        out, _ = self.cell(x)

        out = out[:, -1, :]
        logits = self.fc(out)

        return logits

# GRU
class VitalGRU(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cell = nn.GRU(input_size=config["input_size"], hidden_size=config["hidden_size"],
                           num_layers=config['n_blocks'], dropout=config["drop_out"],
                           batch_first=True)
        self.fc = nn.Linear(in_features=config["hidden_size"], out_features=config["n_classes"], bias=True)

    def forward(self, x):
        out, _ = self.cell(x)

        out = out[:, -1, :]
        logits = self.fc(out)

        return logits


class VitalRCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lstm = nn.LSTM(input_size=config["input_size"], hidden_size=config["hidden_size"],
                            num_layers=config["n_blocks"], dropout=config["drop_out"], bidirectional=True,
                            batch_first=True
                            )
        # self.pool = nn.MaxPool1d(config["max_len"])
        # self.dropout = nn.Dropout(config["drop_out"])
        self.fc = nn.Linear(in_features=config["hidden_size"] * 2 + config["input_size"],
                            out_features=config["n_classes"], bias=True)


    def forward(self, x):

        out, _ = self.lstm(x) # [batch_size, seq_len, hidden_size * 2]
        out = torch.cat((out, x), dim=2) # [batch_size, seq_len, hidden_size * 2 + embed_dim]
        out = F.relu(out) # [batch_size, seq_len, hidden_size * 2 + embed_dim]

        # 在时间步维度做max pooling (列池化)
        out = out.permute(0, 2, 1) # [batch_size, hidden_size * 2 + embed_dim, seq_len]
        # ------------
        # global maxpooling
        out = F.max_pool1d(out, out.size(2)).squeeze(2) # [batch_size, hidden_size * 2 + embed_dim]
        #out = self.pool(out)
        # ------------
        logits = self.fc(out)
        return logits


class VitalAttRNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lstm = nn.LSTM(input_size=config["input_size"], hidden_size=config["hidden_size"],
                            num_layers=config["n_blocks"], dropout=config["drop_out"], bidirectional=True,
                            batch_first=True
                            )
        self.W = nn.Parameter(torch.randn(config["hidden_size"]* 2))
        self.fc = nn.Linear(config["hidden_size"]* 2, config["n_classes"], bias=True)
        # self.dropout = nn.Dropout(config["dropout"])

    def forward(self, x):
        # input : [batch_size, seq_len, embed_size]
        out, _ = self.lstm(x) # [batch_size, seq_len, hidden_size * 2]
        x = torch.tanh(out) # [batch_size, seq_len, hidden_size * 2]
        # attention
        alpha = F.softmax(torch.matmul(x, self.W), dim=1).unsqueeze(-1) # [batch_size, seq_len, 1]
        out = x * alpha #  [ batch_size, seq_len, hidden_size*2]
        out = torch.sum(out, dim=1)  # [ batch_size, hidden_size*2]
        out = F.relu(out)
        logits = self.fc(out)
        return logits
