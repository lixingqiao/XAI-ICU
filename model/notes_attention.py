from torch import nn
import torch
import sys
from .xai_transformer import BertSelfOutput, AttentionBlock, BertPooler

root_dir = './../'
sys.path.append(root_dir)
from .layer_norm import LayerNorm


class NoteSelfAttention(nn.Module):
    def __init__(self, config, embeddings):
        super().__init__()

        n_blocks = config["n_blocks"]
        self.n_blocks = n_blocks
        self.embeds = embeddings

        self.config = config
        self.attention_layers = torch.nn.Sequential(*[AttentionBlock(config) for i in range(n_blocks)])
        # self.output = BertSelfOutput(config)

        self.pooler = BertPooler(config)
        self.classifier = nn.Linear(in_features=config["hidden_size"], out_features=config["n_classes"], bias=True)
        # self.device = config["device"]

        self.attention_probs = {i: [] for i in range(n_blocks)}
        self.attention_debug = {i: [] for i in range(n_blocks)}
        self.attention_gradients = {i: [] for i in range(n_blocks)}
        self.attention_cams = {i: [] for i in range(n_blocks)}

        self.attention_lrp_gradients = {i: [] for i in range(n_blocks)}

    def forward(self, input):

        attn_input = self.embeds(input)

        for i, block in enumerate(self.attention_layers):
            output, attention_probs = block(attn_input)

            self.attention_probs[i] = attention_probs
            #  self.attention_debug[i] = debug_data +  (output,)
            attn_input = output

        pooled = self.pooler(output)
        logits = self.classifier(pooled)

        self.output_ = output
        self.pooled_ = pooled
        self.logits_ = logits

        return logits

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

        # input_states.to(self.config["device"])
        attn_input = self.embeds(input)

        A['input_states'] = attn_input

        for i, block in enumerate(self.attention_layers):
            # [1, 12, 768] -> [1, 12, 768]
            attn_inputdata = attn_input.data
            attn_inputdata.requires_grad_(True)

            A['attn_input_{}_data'.format(i)] = attn_inputdata
            A['attn_input_{}'.format(i)] = attn_input

            gamma = 0. if gammas is None else gammas[i]
            #  print('using gamma', gamma)

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

        all_R = torch.squeeze(R_).detach().cpu().numpy()
        R = R_.sum(2).detach().cpu().numpy()

        if labels is not None:
            loss = torch.nn.CrossEntropyLoss()(logits, labels)
        else:
            loss = None

        return {'loss': loss, 'logits': logits, 'R': R, 'all_R': all_R}

    def compute_with_embedding_input(self, attn_input):

        for i, block in enumerate(self.attention_layers):
            output, attention_probs = block(attn_input)

            self.attention_probs[i] = attention_probs
            #  self.attention_debug[i] = debug_data +  (output,)
            attn_input = output

        pooled = self.pooler(output)
        logits = self.classifier(pooled)

        self.output_ = output
        self.pooled_ = pooled
        self.logits_ = logits

        return logits
