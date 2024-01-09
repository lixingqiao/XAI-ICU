from torch import nn
import torch
import sys
from .xai_transformer import BertSelfOutput, AttentionBlock, BertPooler, PositionalEncoding

root_dir = './../'
sys.path.append(root_dir)
from .layer_norm import LayerNorm


class SubmodalPhysi(nn.Module):
    def __init__(self, config):
        super().__init__()

        n_blocks = config["n_blocks"]
        self.n_blocks = n_blocks

        self.config = config
        self.pos_emb = PositionalEncoding(config)

        self.attention_layers = torch.nn.Sequential(*[AttentionBlock(config) for i in range(n_blocks)])

        self.output = BertSelfOutput(config)

        self.pooler = BertPooler(config)

        self.attention_probs = {i: [] for i in range(n_blocks)}
        self.attention_debug = {i: [] for i in range(n_blocks)}
        self.attention_gradients = {i: [] for i in range(n_blocks)}
        self.attention_cams = {i: [] for i in range(n_blocks)}

        self.attention_lrp_gradients = {i: [] for i in range(n_blocks)}

    def forward(self, input_states):
        # input_states.to(self.config["device"])
        attn_input = self.pos_emb(input_states.transpose(0, 1)).transpose(0, 1)

        for i, block in enumerate(self.attention_layers):
            output, attention_probs = block(attn_input)

            self.attention_probs[i] = attention_probs

            attn_input = output

        pooled = self.pooler(output)
        return pooled

    def xai_forward(self, input_states,
                labels=None,
                gammas=None,
                method=None):

        # Forward
        self.A = {}

        # input_states.to(self.config["device"])
        attn_input = self.pos_emb(input_states.transpose(0, 1)).transpose(0, 1)

        self.A['input_states'] = attn_input

        for i, block in enumerate(self.attention_layers):
            # [1, 12, 768] -> [1, 12, 768]
            attn_inputdata = attn_input.data
            attn_inputdata.requires_grad_(True)

            self.A['attn_input_{}_data'.format(i)] = attn_inputdata
            self.A['attn_input_{}'.format(i)] = attn_input

            gamma = 0. if gammas is None else gammas[i]
            #  print('using gamma', gamma)

            output, attention_probs = block(self.A['attn_input_{}_data'.format(i)], gamma=gamma, method=method)

            self.attention_probs[i] = attention_probs
            attn_input = output

        # (1, 12, 768) -> (1x768)

        outputdata = output.data
        outputdata.requires_grad_(True)

        self.A['output'] = output
        self.A['outputdata'] = outputdata

        pooled = self.pooler(outputdata)
        pooleddata = pooled.data

        self.A['pooled'] = pooled

        pooleddata.requires_grad_(True)
        return pooleddata

    def xai_backward(self, pooleddata, method=None):
        
        (pooleddata.grad * self.A['pooled']).sum().backward()

        Rpool = (self.A['outputdata'].grad * self.A['output'])

        R_ = Rpool
        for i, block in list(enumerate(self.attention_layers))[::-1]:
            R_.sum().backward()

            R_grad = self.A['attn_input_{}_data'.format(i)].grad
            R_attn = R_grad * self.A['attn_input_{}'.format(i)]
            if method == 'GAE':
                self.attention_gradients[i] = block.get_attn_gradients().squeeze()
            R_ = R_attn

        all_R = torch.squeeze(R_).detach().cpu().numpy()
        R = R_.sum(2).detach().cpu().numpy()

        return {'R': R, 'all_R': all_R}


class SubmodalNotes(nn.Module):
    def __init__(self, config, embeddings):
        super().__init__()

        n_blocks = config["n_blocks"]
        self.n_blocks = n_blocks
        self.embeds = embeddings

        self.config = config
        self.attention_layers = torch.nn.Sequential(*[AttentionBlock(config) for i in range(n_blocks)])
        self.pooler = BertPooler(config)

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

            attn_input = output

        pooled = self.pooler(output)

        return pooled


    def xai_forward(self, input_states,
                labels=None,
                gammas=None,
                method=None):

        # Forward
        self.A = {}

        # input_states.to(self.config["device"])
        attn_input = self.embeds(input_states)

        self.A['input_states'] = attn_input

        for i, block in enumerate(self.attention_layers):
            # [1, 12, 768] -> [1, 12, 768]
            attn_inputdata = attn_input.data
            attn_inputdata.requires_grad_(True)

            self.A['attn_input_{}_data'.format(i)] = attn_inputdata
            self.A['attn_input_{}'.format(i)] = attn_input

            gamma = 0. if gammas is None else gammas[i]
            #  print('using gamma', gamma)

            output, attention_probs = block(self.A['attn_input_{}_data'.format(i)], gamma=gamma, method=method)

            self.attention_probs[i] = attention_probs
            attn_input = output

        # (1, 12, 768) -> (1x768)

        outputdata = output.data
        outputdata.requires_grad_(True)

        self.A['output'] = output
        self.A['outputdata'] = outputdata

        pooled = self.pooler(outputdata)
        pooleddata = pooled.data

        self.A['pooled'] = pooled

        pooleddata.requires_grad_(True)
        return pooleddata

    def xai_backward(self, pooleddata, method=None):

        (pooleddata.grad * self.A['pooled']).sum().backward()

        Rpool = (self.A['outputdata'].grad * self.A['output'])

        R_ = Rpool
        for i, block in list(enumerate(self.attention_layers))[::-1]:
            R_.sum().backward()

            R_grad = self.A['attn_input_{}_data'.format(i)].grad
            R_attn = R_grad * self.A['attn_input_{}'.format(i)]
            if method == 'GAE':
                self.attention_gradients[i] = block.get_attn_gradients().squeeze()
            R_ = R_attn

        all_R = torch.squeeze(R_).detach().cpu().numpy()
        R = R_.sum(2).detach().cpu().numpy()

        return {'R': R, 'all_R': all_R}


class SubmodalVital(nn.Module):
    def __init__(self, config):
        super().__init__()

        n_blocks = config["n_blocks"]
        self.n_blocks = n_blocks

        self.config = config

        self.input_projection = nn.Linear(in_features=config['input_size'], out_features=config['hidden_size'])

        self.pos_emb = PositionalEncoding(config)

        self.attention_layers = torch.nn.Sequential(*[AttentionBlock(config) for i in range(n_blocks)])

        self.pooler = BertPooler(config)

        self.attention_probs = {i: [] for i in range(n_blocks)}
        self.attention_debug = {i: [] for i in range(n_blocks)}
        self.attention_gradients = {i: [] for i in range(n_blocks)}
        self.attention_cams = {i: [] for i in range(n_blocks)}

        self.attention_lrp_gradients = {i: [] for i in range(n_blocks)}

    def forward(self, input_states):

        input_states = self.input_projection(input)
        attn_input = self.pos_emb(input_states.transpose(0, 1)).transpose(0, 1)

        for i, block in enumerate(self.attention_layers):

            output, attention_probs = block(attn_input)

            self.attention_probs[i] = attention_probs

            attn_input = output

        pooled = self.pooler(output)
        return pooled


    def xai_forward(self, input,
                labels=None,
                gammas=None,
                method=None):

        # Forward
        self.A = {}

        inputdata = input.data
        inputdata.requires_grad_(True)

        input_states = self.input_projection(inputdata)

        self.A['input'] = input
        self.A['inputdata'] = inputdata

        # input_states.to(self.config["device"])
        attn_input = self.pos_emb(input_states.transpose(0, 1)).transpose(0, 1)

        self.A['input_states'] = attn_input

        for i, block in enumerate(self.attention_layers):
            # [1, 12, 768] -> [1, 12, 768]
            attn_inputdata = attn_input.data
            attn_inputdata.requires_grad_(True)

            self.A['attn_input_{}_data'.format(i)] = attn_inputdata
            self.A['attn_input_{}'.format(i)] = attn_input

            gamma = 0. if gammas is None else gammas[i]
            #  print('using gamma', gamma)

            output, attention_probs = block(self.A['attn_input_{}_data'.format(i)], gamma=gamma, method=method)

            self.attention_probs[i] = attention_probs
            attn_input = output

        # (1, 12, 768) -> (1x768)

        outputdata = output.data
        outputdata.requires_grad_(True)

        self.A['output'] = output
        self.A['outputdata'] = outputdata

        pooled = self.pooler(outputdata)
        pooleddata = pooled.data

        self.A['pooled'] = pooled

        pooleddata.requires_grad_(True)
        return pooleddata

    def xai_backward(self, pooleddata, method=None):

        (pooleddata.grad * self.A['pooled']).sum().backward()

        Rpool = (self.A['outputdata'].grad * self.A['output'])

        R_ = Rpool
        for i, block in list(enumerate(self.attention_layers))[::-1]:
            R_.sum().backward()

            R_grad = self.A['attn_input_{}_data'.format(i)].grad
            R_attn = R_grad * self.A['attn_input_{}'.format(i)]
            if method == 'GAE':
                self.attention_gradients[i] = block.get_attn_gradients().squeeze()
            R_ = R_attn

        R_.sum().backward()
        R_ = self.A['inputdata'].grad * self.A['input']

        all_R = torch.squeeze(R_).detach().cpu().numpy()
        R = R_.sum(2).detach().cpu().numpy()

        return {'R': R, 'all_R': all_R}


class AllModel(nn.Module):
    def __init__(self, config, notes_module=None, vital_module=None, physi_module=None):
        super().__init__()
        self.notes = notes_module
        self.vital = vital_module
        self.physi = physi_module

        self.notes_linear = nn.Linear(in_features=768, out_features=config["notes_hidden"], bias=True)
        self.notes_activation = nn.ReLU()
        self.physi_linear = nn.Linear(in_features=76, out_features=config["physi_hidden"], bias=True)
        self.physi_activation = nn.ReLU()
        self.vital_linear = nn.Linear(in_features=128, out_features=config["vital_hidden"], bias=True)
        self.vital_activation = nn.ReLU()
        self.classifier = nn.Linear(in_features=config["total_hidden"], out_features=config["n_classes"], bias=True)

    def forward(self, notes=None, physi=None, vital=None):
        notes_out = self.notes_activation(self.notes_linear(self.notes(notes)))
        physi_out = self.physi_activation(self.physi_linear(self.physi(physi)))
        vital_out = self.vital_activation(self.vital_linear(self.vital(vital)))

        fusion_out = torch.concat([notes_out, physi_out, vital_out], dim=-1)
        final_out = self.classifier(fusion_out)
        return final_out

    def forward_and_explain(self, notes, physi, vital, cl, method=None):
        # Forward
        physi_pooleddata = self.physi.xai_forward(physi)
        notes_pooleddata = self.notes.xai_forward(notes)
        vital_pooleddata = self.vital.xai_forward(vital)

        notes_out = self.notes_activation(self.notes_linear(notes_pooleddata))
        physi_out = self.physi_activation(self.physi_linear(physi_pooleddata))
        vital_out = self.vital_activation(self.vital_linear(vital_pooleddata))

        fusion_out = torch.concat([notes_out, physi_out, vital_out], dim=-1)

        final_out = self.classifier(fusion_out)

        logits = final_out.detach().cpu().numpy()

        # Explain
        R_final_out = final_out[:, cl]

        assert not physi_pooleddata.grad
        assert not notes_pooleddata.grad
        assert not vital_pooleddata.grad

        R_final_out.backward()

        physi_attribution = self.physi.xai_backward(physi_pooleddata, method=method)
        notes_attribution = self.notes.xai_backward(notes_pooleddata, method=method)
        vital_attribution = self.vital.xai_backward(vital_pooleddata, method=method)

        return logits, physi_attribution, notes_attribution, vital_attribution