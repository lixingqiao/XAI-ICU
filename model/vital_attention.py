from torch import nn
import torch
import sys
from .xai_transformer import BertSelfOutput, AttentionBlock, BertPooler, PositionalEncoding

root_dir = './../'
sys.path.append(root_dir)
from .layer_norm import LayerNorm
import torch.nn.functional as F


class VitalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        n_blocks = config["n_blocks"]
        self.n_blocks = n_blocks

        self.config = config

        self.input_projection = nn.Linear(in_features=config['input_size'], out_features=config['hidden_size'])
        # self.input_projection = nn.Conv1d(in_channels=config['input_size'], out_channels=config['hidden_size'],kernel_size=1)
        self.pos_emb = PositionalEncoding(config)

        self.attention_layers = torch.nn.Sequential(*[AttentionBlock(config) for i in range(n_blocks)])
        # 定义了但是没用上？

        self.pooler = BertPooler(config)
        self.classifier = nn.Linear(in_features=config["hidden_size"], out_features=config["n_classes"], bias=True)
        # self.device = config["device"]

        self.attention_probs = {i: [] for i in range(n_blocks)}
        self.attention_debug = {i: [] for i in range(n_blocks)}
        self.attention_gradients = {i: [] for i in range(n_blocks)}
        self.attention_cams = {i: [] for i in range(n_blocks)}

        self.attention_lrp_gradients = {i: [] for i in range(n_blocks)}

    def forward(self, input):

        # input_states = self.input_projection(input.permute(0, 2, 1)).permute(0, 2, 1)
        input_states = self.input_projection(input)
        attn_input = self.pos_emb(input_states.transpose(0, 1)).transpose(0, 1)

        for i, block in enumerate(self.attention_layers):

            output, attention_probs = block(attn_input)

            self.attention_probs[i] = attention_probs

            attn_input = output
        # ----
        pooled = self.pooler(output)
        logits = self.classifier(pooled)

        self.output_ = output
        self.pooled_ = pooled
        self.logits_ = logits

        return logits
        # ----
        # output = output.permute(0, 2, 1)
        # output = F.max_pool1d(output, output.size(2)).squeeze(2)
        # logits = self.classifier(output)
        #
        # return logits

    def prep_lrp(self, x):
        x = x.data
        x.requires_grad_(True)
        return x

    def forward_and_explain(self, input,
                            cl,
                            labels=None,
                            gammas=None,
                            method=None):

        # Forward
        A = {}
        # linear proj
        inputdata = input.data
        inputdata.requires_grad_(True)

        input_states = self.input_projection(inputdata)

        attn_input = self.pos_emb(input_states.transpose(0, 1)).transpose(0, 1)

        A['input_states'] = attn_input

        for i, block in enumerate(self.attention_layers):
            # [1, 12, 768] -> [1, 12, 768]
            attn_inputdata = attn_input.data
            attn_inputdata.requires_grad_(True)

            A['attn_input_{}_data'.format(i)] = attn_inputdata
            A['attn_input_{}'.format(i)] = attn_input

            gamma = 0. if gammas is None else gammas[i]

            output, attention_probs = block(A['attn_input_{}_data'.format(i)], gamma=gamma, method=method)

            self.attention_probs[i] = attention_probs

            attn_input = output

        # (1, 12, 768) -> (1x768)

        outputdata = output.data
        outputdata.requires_grad_(True)

        pooled = self.pooler(outputdata)  # A['attn_output'] )

        # (1x768) -> (1,nclasses)
        pooleddata = pooled.data
        pooleddata.requires_grad_(True)

        logits = self.classifier(pooleddata)

        A['logits'] = logits

        # Through clf layer
        Rout = A['logits'][:, cl]

        self.R0 = Rout.detach().cpu().numpy()

        Rout.backward()
        (pooleddata.grad * pooled).sum().backward()

        Rpool = (outputdata.grad * output)

        R_ = Rpool
        for i, block in list(enumerate(self.attention_layers))[::-1]:
            R_.sum().backward()

            R_grad = A['attn_input_{}_data'.format(i)].grad
            R_attn = R_grad * A['attn_input_{}'.format(i)]
            if method == 'GAE':
                self.attention_gradients[i] = block.get_attn_gradients().squeeze()
            R_ = R_attn

        # linear project
        R_.sum().backward()
        R_ = inputdata.grad * input

        all_R = torch.squeeze(R_).detach().cpu().numpy()
        R = R_.sum(2).detach().cpu().numpy()

        if labels is not None:
            loss = torch.nn.CrossEntropyLoss()(logits, labels)
        else:
            loss = None

        return {'loss': loss, 'logits': logits, 'R': R, 'all_R': all_R}
