import sys
import os

basedir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(basedir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import warnings

warnings.filterwarnings("ignore")

from transformers import AutoTokenizer, AutoModel
import transformers

transformers.logging.set_verbosity_error()

import pickle
import numpy as np

from torch import nn
import torch

from model import BertSelfOutput, AttentionBlock, BertPooler, NoteSelfAttention, LayerNorm
from utils import set_seed, get_config_from_json, get_args
from utils.attribution import _compute_rollout_attention, compute_joint_attention, _compute_integrated_gradient_nlp


set_seed(1234)


def softmax(x):
    '''Compute softmax values for each sets of scores in x.'''
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def flip(model, attribution, data, y_true, fracs, tokenizer, flip_case, random_order = False,  device='cpu'):

    attribution = np.array(attribution)

    UNK_IDX = tokenizer.unk_token_id

    inputs0 = torch.tensor(data).to(device)

    y0 = model(inputs0).squeeze().detach().cpu().numpy()

    if random_order == False:
        if flip_case == 'generate':
            # big to small
            inds_sorted = np.argsort(attribution)[::-1]
        elif flip_case == 'pruning':
            # small to big
            inds_sorted = np.argsort(np.abs(attribution))
        else:
            raise
    else:
        inds_ = np.array(list(range(attribution.shape[-1])))
        remain_inds = np.array(inds_)
        np.random.shuffle(remain_inds)

        inds_sorted = remain_inds

    inds_sorted = inds_sorted.copy()
    vals = attribution[inds_sorted]

    mse = []
    evidence = []
    model_outs = {'y_true': y_true.detach().cpu().numpy(), 'y0': y0}

    N = len(attribution)

    evolution = {}
    for frac in fracs:
        inds_generator = iter(inds_sorted)
        n_flip = int(np.ceil(frac * N))
        inds_flip = [next(inds_generator) for i in range(n_flip)]

        if flip_case == 'pruning':

            inputs = inputs0

            for i in inds_flip:

                inputs[:, i] = UNK_IDX

        elif flip_case == 'generate':

            inputs = UNK_IDX * torch.ones_like(inputs0)

            for i in inds_flip:

                inputs[:, i] = inputs0[:, i]

        y = model(inputs).detach().cpu().numpy()

        y = y.squeeze()

        err = np.sum((y0 - y) ** 2)

        mse.append(err)

        evidence.append(softmax(y))

        #  print('{:0.2f}'.format(frac), ' '.join(tokenizer.convert_ids_to_tokens(inputs.detach().cpu().numpy().squeeze())))
        evolution[frac] = (inputs.detach().cpu().numpy(), inds_flip, y)

    if flip_case == 'generate' and frac == 1.:
        assert (inputs0 == inputs).all()

    model_outs['flip_evolution'] = evolution
    return mse, evidence, model_outs


def run(config):
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    # mask token for physi
    # physi mask are generate by the default values in utils/physi_discretizer_config.json via discretizer and normalizer)
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    bert_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    # the grads in embedding are not fix
    pretrained_embeds = bert_model.embeddings
    pretrained_embeds.to(device)

    for name, parameters in pretrained_embeds.named_parameters():
        parameters.requires_grad = False

    for fold_i in range(5):
        with open(f'data/dataset/pickle/align_notes/fold_{fold_i}_note.dat', 'rb') as f:
            data_dict = pickle.load(f)
        test_data = data_dict['test']

        print(f"Begin to test in {fold_i}")

        note_params = torch.load(os.path.join(config['path_setting']['notes_ckpt_path'], f"best_model_state_{fold_i}.bin")).state_dict()

        note_config = config['model_setting']['notes']

        # set model for explain
        note_config["detach_layernorm"] = False  # Detaches the attention-block-output LayerNorm
        note_config["detach_kq"] = False
        model = NoteSelfAttention(note_config, pretrained_embeds)
        model.load_state_dict(note_params)
        model.to(device)
        models = {'none': model}

        # Transformer Model
        note_config["detach_layernorm"] = False  # Detaches the attention-block-output LayerNorm
        note_config["detach_kq"] = True
        model = NoteSelfAttention(note_config, pretrained_embeds)
        model.load_state_dict(note_params)
        model.to(device)
        models['detach_KQ'] = model

        print('Detach SM +  LN')
        # Transformer Model
        note_config["detach_layernorm"] = True  # Detaches the attention-block-output LayerNorm
        note_config["detach_kq"] = True
        model = NoteSelfAttention(note_config, pretrained_embeds)
        model.load_state_dict(note_params)
        model.to(device)
        models['detach_KQ_LNorm'] = model

        print('Detach SM +  LN without mean')
        # Transformer Model
        note_config["detach_layernorm"] = True  # Detaches the attention-block-output LayerNorm
        note_config["detach_mean"] = False  # Detaches the attention-block-output LayerNorm
        note_config["detach_kq"] = True
        model = NoteSelfAttention(note_config, pretrained_embeds)
        model.load_state_dict(note_params)
        model.to(device)
        models['detach_KQ_LNorm_Norm'] = model

        fracs = np.linspace(0, 1., 11)

        # for flip_case in ['generate', 'pruning']:
        for flip_case in ['pruning']:
            print(flip_case)
            all_flips = {}

            res_conservation = {}
            for case, random_order in [#('random', True),
                                       #('attn_last', False),
                                       #('rollout_2', False),
                                       # ('GAE', False),
                                       #('gi', False),
                                       #('lrp_detach_KQ', False),
                                       #('lrp_detach_KQ_LNorm_Norm', False),
                                        ('ig', False)
                                       ]:

                print(case)
                layer_idxs = model.attention_probs.keys()
                M, E, EVOLUTION = [], [], []
                C = []

                j = 0

                if case in ['gi', 'lrp', 'GAE', 'ig']:
                    model = models['none']
                elif case in ['gi_detach_KQ', 'lrp_detach_KQ']:
                    model = models['detach_KQ']
                elif case in ['detach_KQ_LNorm_Norm', 'lrp_detach_KQ_LNorm_Norm']:
                    model = models['detach_KQ_LNorm_Norm']
                else:
                    model = models['detach_KQ_LNorm']

                total_test_num = len(test_data)

                for i in range(total_test_num):
                    x = test_data[i]

                    input_tensor = torch.tensor(np.float32(x["input_ids"]), requires_grad=True).unsqueeze(0).long().to(device)

                    y_true = torch.tensor(x['mortality']).to(device)
                    if case == 'ig':

                        embedding_data = model.embeds(input_tensor)

                        inputs_baseline = torch.tensor(torch.randint(0, 28995, size=input_tensor.shape)).long().to(device)
                        embedding_baseline = model.embeds(inputs_baseline)

                        attribution = _compute_integrated_gradient_nlp(embedding_data, embedding_baseline, model, y_true)

                    if case == 'GAE':
                        outs = model.forward_and_explain(input_tensor, cl=y_true, method=case)
                    else:
                        outs = model(input_tensor)

                    if case == 'random':
                        attribution = np.random.normal(0, 1, input_tensor.shape[1])

                    elif case == 'attn_last':
                        attribution = np.mean(
                            [x_.sum(0) for x_ in model.attention_probs[max(layer_idxs)].detach().cpu().numpy()[0]], 0)
                    # add rollout here
                    elif 'rollout' in case:
                        attns = [model.attention_probs[k].detach().cpu().numpy() for k in
                                 sorted(model.attention_probs.keys())]
                        attentions_mat = np.stack(attns, axis=0).squeeze()
                        res_att_mat = attentions_mat.sum(axis=1) / attentions_mat.shape[1]
                        joint_attentions = compute_joint_attention(res_att_mat, add_residual=True)
                        # first layer in joint_attentions is right, the original code is wrong.
                        # idx = int(case.replace('rollout_', ''))
                        attribution = joint_attentions[min(layer_idxs)].sum(0)
                    elif 'GAE' in case:
                        attns = [model.attention_probs[k].detach().cpu().numpy() for k in
                                 sorted(model.attention_probs.keys())]
                        attentions_mat = np.stack(attns, axis=0).squeeze()
                        attns = [model.attention_gradients[k].detach().cpu().numpy() for k in
                                 sorted(model.attention_gradients.keys())]
                        attentions_grads = np.stack(attns, axis=0).squeeze()
                        attentions_mat = torch.tensor(attentions_mat * attentions_grads).clamp(min=0)
                        attentions_mat = torch.tensor(attentions_mat).clamp(min=0).mean(dim=1)
                        joint_attentions = _compute_rollout_attention(attentions_mat)
                        joint_attentions[:, 0, 0] = 0
                        idx = 0
                        attribution = joint_attentions[idx].sum(0)

                    elif case in ['gi', 'gi_detach_KQ_LNorm', 'gi_detach_KQ', 'gi_detach_KQ_LNorm_Norm']:
                        outs = model.forward_and_explain(input_tensor, cl=y_true,
                                                         # abels=labels_in,
                                                         gammas=model.n_blocks*[0.0])

                        attribution = outs['R'].squeeze()


                    elif case in ['lrp', 'lrp_detach_KQ_LNorm', 'lrp_detach_KQ', 'lrp_detach_KQ_LNorm_Norm']:

                        outs = model.forward_and_explain(input_tensor, cl=y_true,
                                                         # abels=labels_in,
                                                         gammas=model.n_blocks*[0.0])

                        attribution = outs['R'].squeeze()

                    m, e, evolution = flip(model,
                                           attribution=attribution,
                                           data=input_tensor,
                                           y_true=y_true,
                                           fracs=fracs,
                                           tokenizer=tokenizer,
                                           flip_case=flip_case,
                                           random_order=random_order,
                                           device=device)

                    # print(e)
                    M.append(m)
                    E.append(e)
                    EVOLUTION.append(evolution)

                    if j % 100 == 0:
                        print('****', j)

                    j += 1

                all_flips[case] = {'E': E, 'M': M, 'Evolution': EVOLUTION}

            pickle.dump(all_flips, open(f'exp_explain_multimodal/note_explain/fold_{fold_i}_all_flips_{flip_case}_ig_50.p', 'wb'))

if __name__ == "__main__":
    args = get_args()
    config = get_config_from_json(args.config)
    save_dir = 'exp_explain_multimodal/note_explain'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    run(config)
