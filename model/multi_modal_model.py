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

    def forward(self, input_states):
        # input_states.to(self.config["device"])
        attn_input = self.pos_emb(input_states.transpose(0, 1)).transpose(0, 1)

        for i, block in enumerate(self.attention_layers):
            output, attention_probs = block(attn_input)

            attn_input = output

        pooled = self.pooler(output)
        return pooled


class SubmodalNotes(nn.Module):
    def __init__(self, config, embeddings):
        super().__init__()

        n_blocks = config["n_blocks"]
        self.n_blocks = n_blocks
        self.embeds = embeddings

        self.config = config
        self.attention_layers = torch.nn.Sequential(*[AttentionBlock(config) for i in range(n_blocks)])
        self.pooler = BertPooler(config)

    def forward(self, input):

        attn_input = self.embeds(input)

        for i, block in enumerate(self.attention_layers):
            output, attention_probs = block(attn_input)
            attn_input = output

        pooled = self.pooler(output)

        return pooled


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

    def forward(self, input):
        # input_states.to(self.config["device"])
        input_states = self.input_projection(input)
        attn_input = self.pos_emb(input_states.transpose(0, 1)).transpose(0, 1)

        for i, block in enumerate(self.attention_layers):

            output, attention_probs = block(attn_input)

            attn_input = output

        pooled = self.pooler(output)

        return pooled


class PhysiNotesModel(nn.Module):
    def __init__(self, config, notes_module=None, physi_module=None):
        super().__init__()
        self.notes = notes_module
        self.physi = physi_module
        # dropout
        self.dropout = torch.nn.Dropout(0.3, inplace=False)

        self.notes_linear = nn.Linear(in_features=768, out_features=config["notes_hidden"], bias=True)
        self.notes_activation = nn.Tanh()
        self.physi_linear = nn.Linear(in_features=76, out_features=config["physi_hidden"], bias=True)
        self.physi_activation = nn.Tanh()

        self.classifier = nn.Linear(in_features=config["total_hidden"], out_features=config["n_classes"], bias=True)

    def forward(self, notes=None, physi=None, vital=None):
        notes_out = self.notes_activation(self.notes_linear(self.notes(notes)))
        physi_out = self.physi_activation(self.physi_linear(self.physi(physi)))

        fusion_out = torch.concat([notes_out, physi_out], dim=-1)

        fusion_out = self.dropout(fusion_out)

        final_out = self.classifier(fusion_out)
        return final_out


class PhysiVitalModel(nn.Module):
    def __init__(self, config, vital_module=None, physi_module=None):
        super().__init__()
        self.vital = vital_module
        self.physi = physi_module

        # dropout
        self.dropout = torch.nn.Dropout(0.3, inplace=False)

        self.vital_linear = nn.Linear(in_features=128, out_features=config["vital_hidden"], bias=True)
        self.vital_activation = nn.Tanh()
        self.physi_linear = nn.Linear(in_features=76, out_features=config["physi_hidden"])
        self.physi_activation = nn.Tanh()
        self.classifier = nn.Linear(in_features=config["total_hidden"], out_features=config["n_classes"], bias=True)

    def forward(self, notes=None, physi=None, vital=None):
        vital_out = self.vital_activation(self.vital_linear(self.vital(vital)))
        physi_out = self.physi_activation(self.physi_linear(self.physi(physi)))

        fusion_out = torch.concat([physi_out, vital_out], dim=-1)

        fusion_out = self.dropout(fusion_out)

        final_out = self.classifier(fusion_out)

        return final_out


class NotesVitalModel(nn.Module):
    def __init__(self, config, notes_module=None, vital_module=None):
        super().__init__()
        self.notes = notes_module
        self.vital = vital_module
        # dropout
        self.dropout = torch.nn.Dropout(0.3, inplace=False)

        self.notes_linear = nn.Linear(in_features=768, out_features=config["notes_hidden"], bias=True)
        self.notes_activation = nn.Tanh()
        self.vital_linear = nn.Linear(in_features=128, out_features=config["vital_hidden"], bias=True)
        self.vital_activation = nn.Tanh()
        self.classifier = nn.Linear(in_features=config["total_hidden"], out_features=config["n_classes"], bias=True)

    def forward(self, notes=None, physi=None, vital=None):
        notes_out = self.notes_activation(self.notes_linear(self.notes(notes)))
        vital_out = self.vital_activation(self.vital_linear(self.vital(vital)))

        fusion_out = torch.concat([notes_out, vital_out], dim=-1)

        fusion_out = self.dropout(fusion_out)

        final_out = self.classifier(fusion_out)
        return final_out


class AllModel(nn.Module):
    def __init__(self, config, notes_module=None, vital_module=None, physi_module=None):
        super().__init__()
        self.notes = notes_module
        self.vital = vital_module
        self.physi = physi_module

        # dropout
        self.dropout = torch.nn.Dropout(0.3, inplace=False)

        self.notes_linear = nn.Linear(in_features=768, out_features=config["notes_hidden"], bias=True)
        self.notes_activation = nn.Tanh()
        self.physi_linear = nn.Linear(in_features=76, out_features=config["physi_hidden"], bias=True)
        self.physi_activation = nn.Tanh()
        self.vital_linear = nn.Linear(in_features=128, out_features=config["vital_hidden"], bias=True)
        self.vital_activation = nn.Tanh()
        self.classifier = nn.Linear(in_features=config["total_hidden"], out_features=config["n_classes"], bias=True)

    def forward(self, notes=None, physi=None, vital=None):
        notes_out = self.notes_activation(self.notes_linear(self.notes(notes)))
        physi_out = self.physi_activation(self.physi_linear(self.physi(physi)))
        vital_out = self.vital_activation(self.vital_linear(self.vital(vital)))

        fusion_out = torch.concat([notes_out, physi_out, vital_out], dim=-1)
        fusion_out = self.dropout(fusion_out)
        final_out = self.classifier(fusion_out)

        return final_out