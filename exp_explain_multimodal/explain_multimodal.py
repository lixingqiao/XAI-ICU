import os
import pickle
import sys

basedir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(basedir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import warnings

warnings.filterwarnings("ignore")

from transformers import AutoTokenizer, AutoModel
import transformers

transformers.logging.set_verbosity_error()

from utils import *

import torch
import numpy as np
from xai_model import SubmodalVital, SubmodalPhysi, SubmodalNotes, AllModel
from utils import set_seed, get_config_from_json, get_args


set_seed(1234)


def run(config):
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    bert_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    ### the grads in embedding are not fix
    pretrained_embeds = bert_model.embeddings
    pretrained_embeds.to(device)

    for name, parameters in pretrained_embeds.named_parameters():
        parameters.requires_grad = False


    for fold_i in range(5):
        with open(f'data/dataset/pickle/align_all/fold_{fold_i}.dat', 'rb') as f:
            data_dict = pickle.load(f)
        test_data = data_dict['test']

        print(f"Begin to test in {fold_i}")

        multimodal_params = torch.load(os.path.join(config['path_setting']['multimodal_ckpt_path'], f"best_model_state_{fold_i}.bin")).state_dict()

        config['model_setting']['physi']['detach_mean'] = False
        config['model_setting']['vital']['detach_mean'] = False
        config['model_setting']['notes']['detach_mean'] = False

        physi_module = SubmodalPhysi(config['model_setting']['physi'])
        vital_module = SubmodalVital(config['model_setting']['vital'])
        notes_module = SubmodalNotes(config['model_setting']['notes'], pretrained_embeds)

        model = AllModel(config['model_setting']['notes_physi_vital'], notes_module=notes_module,
                         vital_module=vital_module, physi_module=physi_module)

        model.load_state_dict(multimodal_params)
        model.to(device)

        cnt = 0

        out_attribution_list = []

        for x in test_data:

            if x['mortality'] != 1:
                continue

            notes_input = torch.tensor(np.float32(x['input_ids']), requires_grad=True).unsqueeze(0).long().to(device)
            physi_input = torch.tensor(np.float32(x['physi']), requires_grad=True).unsqueeze(0).to(device)
            vital_input = torch.tensor(np.float32(x['vital']), requires_grad=True).unsqueeze(0).to(device)

            y_true = torch.tensor(x['mortality']).to(device)

            # target = torch.tensor(np.float32(x['mortality'])).unsqueeze(0).long()

            assert notes_input.shape == (1, 512)
            assert physi_input.shape == (1, 24, 76)
            assert vital_input.shape == (1, 480, 21)

            # outputs = model(notes=notes_input, physi=physi_input, vital=vital_input)

            outputs_attr = model.forward_and_explain(notes=notes_input, physi=physi_input, vital=vital_input, cl=y_true)

            out_attribution_list.append([outputs_attr, notes_input, physi_input, vital_input])

            cnt += 1
            if cnt % 10 == 0:
                print('****', cnt)
        pickle.dump(out_attribution_list, open(f'exp_explain_multimodal/multimodal_explain/fold_{fold_i}.p', 'wb'))


if __name__ == "__main__":
    args = get_args()
    config = get_config_from_json(args.config)
    save_dir = 'exp_explain_multimodal/multimodal_explain'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    run(config)