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


# 标准RNN模型
class PhysiRNN(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cell = nn.RNN(input_size=config["input_size"], hidden_size=config["hidden_size"],
                           num_layers=config['n_blocks'], dropout=config["drop_out"],
                           nonlinearity=config['nonlinearity'], batch_first=True)
        self.fc = nn.Linear(in_features=config["hidden_size"], out_features=config["n_classes"], bias=True)

    def forward(self, x):
        # input shape: (batch_size, seq_len, dim)
        batch_size = x.size(0)
        h0 = Variable(torch.zeros(self.config['n_blocks'] * 1, batch_size, self.config["hidden_size"]))
        h0 = h0.cuda()

        out, _ = self.cell(x, h0)
        # out shape: (batch_size, seq_len, hidden_size)
        out = out[:, -1, :]
        # out shape: （bach_size, hidden_size）
        logits = self.fc(out)

        return logits


# LSTM
class PhysiLSTM(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cell = nn.LSTM(input_size=config["input_size"], hidden_size=config["hidden_size"],
                            num_layers=config['n_blocks'], dropout=config["drop_out"],
                            batch_first=True)
        self.fc = nn.Linear(in_features=config["hidden_size"], out_features=config["n_classes"], bias=True)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = Variable(torch.zeros(self.config['n_blocks'] * 1, batch_size, self.config["hidden_size"]))
        c0 = Variable(torch.zeros(self.config['n_blocks'] * 1, batch_size, self.config["hidden_size"]))

        h0 = h0.cuda()
        c0 = c0.cuda()
        out, _ = self.cell(x, (h0, c0))

        out = out[:, -1, :]
        logits = self.fc(out)

        return logits

# GRU
class PhysiGRU(nn.Module):

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


class PhysiRCNN(nn.Module):
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


class PhysiAttRNN(nn.Module):
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


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        self.first_token_tensor = hidden_states[:, 0]
        self.pooled_output1 = self.dense(self.first_token_tensor)
        self.pooled_output2 = self.activation(self.pooled_output1)
        return self.pooled_output2


class PhysiTransformer(nn.Module):
    """
    Input:
        X: (n_samples, n_length, n_channel)
    Output:
        out: (n_samples, n_classes)
    """
    def __init__(self, config):
        super().__init__()
        self.d_model = config["hidden_size"]
        self.nhead = config["num_attention_heads"]
        self.dim_feedforward = config["d_ffn"]
        self.dropout = config["drop_out"]
        self.n_classes = config["n_classes"]
        self.activation = config["activation"]

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config["n_blocks"])
        self.pooler = BertPooler(config)
        self.classifier = nn.Linear(in_features=config["hidden_size"], out_features=config["n_classes"], bias=True)

    def forward(self, x):

        out = x

        out = self.transformer_encoder(out) # [batch_size, seq_len, hidden_size]

        # ------------
        # method 1
        # global maxpooling
        # out = out.permute(0, 2, 1)  # [batch_size, hidden_size, seq_len]
        # out = F.max_pool1d(out, out.size(2)).squeeze(2) # [batch_size, hidden_size]

        # ------------
        # method 2
        # pooler
        out = self.pooler(out)

        out = self.classifier(out)

        return out