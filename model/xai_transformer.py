import copy

import math
import torch
import torch.nn as nn

from model.layer_norm import LayerNorm


class LNargs(object):

    def __init__(self):
        self.lnv = 'nowb'
        self.sigma = None
        self.hidden = None
        self.adanorm_scale = 1.
        self.nowb_scale = None
        self.mean_detach = False
        self.std_detach = False


class LNargsDetach(object):

    def __init__(self):
        self.lnv = 'nowb'
        self.sigma = None
        self.hidden = None
        self.adanorm_scale = 1.
        self.nowb_scale = None
        self.mean_detach = True
        self.std_detach = True


class LNargsDetachNotMean(object):

    def __init__(self):
        self.lnv = 'nowb'
        self.sigma = None
        self.hidden = None
        self.adanorm_scale = 1.
        self.nowb_scale = None
        self.mean_detach = False
        self.std_detach = True


def make_p_layer(layer, gamma):
    player = copy.deepcopy(layer)
    player.weight = torch.nn.Parameter(layer.weight + gamma * layer.weight.clamp(min=0))
    player.bias = torch.nn.Parameter(layer.bias + gamma * layer.bias.clamp(min=0))
    return player


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the utils by simply taking the hidden state corresponding
        # to the first token.
        self.first_token_tensor = hidden_states[:, 0]
        self.pooled_output1 = self.dense(self.first_token_tensor)
        self.pooled_output2 = self.activation(self.pooled_output1)
        return self.pooled_output2


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config["all_head_size"], config["hidden_size"])
        self.config = config

        if self.config["train_mode"] == True:
            self.dropout = torch.nn.Dropout(config["drop_out"], inplace=False)

        if config["detach_layernorm"] == True:
            assert config["train_mode"] == False

            if config["detach_mean"] == False:
                print('Detach LayerNorm only Norm')
                largs = LNargsDetachNotMean()
            else:
                print('Detach LayerNorm Mean+Norm')
                largs = LNargsDetach()
        else:
            largs = LNargs()

        self.LayerNorm = LayerNorm(config["hidden_size"], eps=config["layer_norm_eps"], args=largs)

        self.detach = config["detach_layernorm"]

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        if self.config["train_mode"] == True:
            hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

    def pforward(self, hidden_states, input_tensor, gamma):
        pdense = make_p_layer(self.dense, gamma)
        hidden_states = pdense(hidden_states)
        # hidden_states = self.dense(hidden_states)
        if self.config["train_mode"] == True:
            hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class AttentionBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.query = nn.Linear(config["hidden_size"], config["all_head_size"])
        self.key = nn.Linear(config["hidden_size"], config["all_head_size"])
        self.value = nn.Linear(config["hidden_size"], config["all_head_size"])
        self.output = BertSelfOutput(config)
        self.detach = config["detach_kq"]
        if self.config["train_mode"] == True:
            self.dropout = torch.nn.Dropout(config["drop_out"], inplace=False)

        if self.detach == True:
            assert self.config["train_mode"] == False
            print('Detach K-Q-softmax branch')

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def transpose_for_scores(self, x):
        # x torch.Size([1, 10, 768])
        # xout torch.Size([1, 10, 12, 64])
        new_x_shape = x.size()[:-1] + (self.config["num_attention_heads"], self.config["attention_head_size"])
        x = x.view(*new_x_shape)
        X = x.permute(0, 2, 1, 3)
        return X

    def un_transpose_for_scores(self, x, old_shape):
        x = x.permute(0, 1, 2, 3)
        return x.reshape(old_shape)

    @staticmethod
    def pproc(layer, player, x):
        eps = 1e-5
        z = layer(x)
        zp = player(x)
        # z / zp , if 0 in zp, the value will be nan. zp()
        return (zp + eps) * (z / (zp + eps)).data

    def forward(self, hidden_states, gamma=0, method=None):

        #  print('PKQ gamma', gamma)

        pquery = make_p_layer(self.query, gamma)
        pkey = make_p_layer(self.key, gamma)
        pvalue = make_p_layer(self.value, gamma)

        if self.config["train_mode"]:
            query_ = self.query(hidden_states)
            key_ = self.key(hidden_states)
            val_ = self.value(hidden_states)
        else:
            query_ = self.pproc(self.query, pquery, hidden_states)
            key_ = self.pproc(self.key, pkey, hidden_states)
            val_ = self.pproc(self.value, pvalue, hidden_states)

        # [1, senlen, 768] -> [1, 12, senlen, 64]
        query_t = self.transpose_for_scores(query_)
        key_t = self.transpose_for_scores(key_)
        val_t = self.transpose_for_scores(val_)

        # torch.Size([1, 12, 10, 64]) , torch.Size([1, 12, 64, 10]) -> torch.Size([1, 12, 10, 10])
        attention_scores = torch.matmul(query_t, key_t.transpose(-1, -2))

        # if torch.isnan(attention_scores).any():
        #    import pdb;pdb.set_trace()

        if self.detach:
            assert self.config["train_mode"] == False
            attention_probs = nn.Softmax(dim=-1)(attention_scores).detach()
        else:
            # 就这一行的区别 LRP_detach, detach 后，不会看 attention score的导数
            attention_probs = nn.Softmax(dim=-1)(attention_scores)

        if self.config["train_mode"]:
            attention_probs = self.dropout(attention_probs)
        if method == 'GAE':
            attention_probs.register_hook(self.save_attn_gradients)

        context_layer = torch.matmul(attention_probs, val_t)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        old_context_layer_shape = context_layer.shape
        new_context_layer_shape = context_layer.size()[:-2] + (self.config["all_head_size"],)
        context_layer = context_layer.view(*new_context_layer_shape)

        if self.config["train_mode"]:
            output = self.output(context_layer, hidden_states)
        else:
            output = self.output.pforward(context_layer, hidden_states, gamma=gamma)
        return output, attention_probs  # , (attention_scores, hidden_states) #, query_t, key_t, val_t)


class PositionalEncoding(nn.Module):
    def __init__(self, config, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        d_model = config["hidden_size"]
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        odd_len = d_model - div_term.size(-1)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:odd_len])

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        self.config = config

    def forward(self, x):
        """
        input & out shape:
            x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]

        if self.config["train_mode"]:
            x = self.dropout(x)
        return x
