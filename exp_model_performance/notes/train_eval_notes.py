import pickle
import shutil
import sys
import os
from datetime import datetime

import numpy as np
import pandas as pd
import warnings

from torch.optim import lr_scheduler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, logging
import transformers

transformers.logging.set_verbosity_error()

warnings.filterwarnings("ignore")

# Add file and basedir to Path
basedir = os.path.split(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))[0]
sys.path.append(basedir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset

from utils import set_seed, get_config_from_json, evaluate_metrics, get_args, EarlyStopping, AUCEarlyStopping
from utils import MultiModalReader, MultiModalDataset

from model import NoteSelfAttention, NoteRNN, NoteLSTM, NoteGRU, NoteRCNN, NoteAttRNN


def train_epoch(model, data_loader, optimizer, device, epoch, num_epochs, n_examples):
    model = model.train()
    current_loss = 0.0
    correct_predictions = 0
    losses = []
    train_bar = tqdm(data_loader)
    for itr, sample in enumerate(train_bar):
        total = len(data_loader)
        # input_data = sample["data"].to(device)
        # targets = sample["label"].to(device)
        input_ids = sample["input_ids"].type(torch.long).to(device)
        targets = sample["mortality"].type(torch.long).to(device)

        outputs = model(input_ids)
        loss = F.cross_entropy(outputs, targets)
        preds = outputs.argmax(dim=1)
        # metric
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        # backward
        optimizer.zero_grad()
        loss.backward()
        # clip grad
        # nn.utils.clip_grad_norm_(model.named_parameters().values(), max_norm=1.0)
        optimizer.step()

        current_loss += loss.item()
        # Standard out
        train_bar.desc = 'Epoch[{}/{}] | Batch[{}/{}], loss: {:.3f}'.format(epoch + 1, num_epochs, itr + 1,
                                                                            total,
                                                                            current_loss / (itr + 1))
    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, device):
    model = model.eval()

    losses = []
    # correct_predictions = 0
    with torch.no_grad():
        # -----
        y_label_flag = 0
        y_score_flag = 0
        # -----
        for itr, sample in enumerate(data_loader):
            input_ids = sample["input_ids"].type(torch.long).to(device)
            targets = sample["mortality"].type(torch.long).to(device)
            outputs = model(input_ids)

            # preds = outputs.argmax(dim=1)
            # correct_predictions += torch.sum(preds == targets)

            y_pred = torch.softmax(outputs, dim=1)[:, 1]
            # -----
            if y_label_flag:
                y_label = torch.cat((y_label, targets), 0)
            else:
                y_label = targets
                y_label_flag = 1
            if y_score_flag:
                y_score = torch.cat((y_score, y_pred), 0)
            else:
                y_score = y_pred
                y_score_flag = 1
            # -----
            loss = F.cross_entropy(outputs, targets)
            losses.append(loss.item())
    # calculate metric
    y_label = y_label.to("cpu").numpy().flatten()
    y_score = y_score.to("cpu").numpy().flatten()
    auc_roc, auc_pr, acc, sensitive, specificty = evaluate_metrics(y_label, y_score)

    return np.mean(losses), auc_roc, auc_pr, acc, sensitive, specificty


def set_save_df():
    df = pd.DataFrame()
    df.insert(0, 'fold', '')
    df.insert(1, 'epoch', '')
    df.insert(2, 'loss', '')
    df.insert(3, 'auc_roc', '')
    df.insert(4, 'auc_pr', '')
    df.insert(5, 'acc', '')
    return df


def load_data(train_df, test_df, n_fold, pickle_path):
    pickle_file = os.path.join(pickle_path, n_fold + "_note" + ".dat")
    print(pickle_file)
    if os.path.exists(pickle_file):
        print("Data pickle already exist...")
        with open(pickle_file, "rb") as f:
            data_dict = pickle.load(f)
    else:
        print("Generate pickles...")
        if not os.path.exists(pickle_path):
            os.makedirs(pickle_path)
        # Read data to dict
        train_reader = MultiModalReader(dataset_dir="data/",
                                        listfile_df=train_df,
                                        period_length=24.0, modal=["notes"])
        test_reader = MultiModalReader(dataset_dir="data/",
                                       listfile_df=test_df,
                                       period_length=24.0, modal=["notes"])
        # Load data
        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        train_dataset = MultiModalDataset.load(train_reader, tokenizer=tokenizer, modals=["notes"])
        test_datset = MultiModalDataset.load(test_reader, tokenizer=tokenizer, modals=["notes"])
        data_dict = {"train": train_dataset, "test": test_datset}
        with open(pickle_file, "wb") as f:
            pickle.dump(data_dict, f)

    return data_dict['train'], data_dict['test']


def train_and_eval(config):
    # Set random seed.
    set_seed(config['exp_setting']['random_seed'])
    # set device.
    device = torch.device(f"cuda:{config['exp_setting']['device_id']}" if torch.cuda.is_available() else "cpu")
    config['exp_setting']['device'] = device
    # save df
    train_info_df = set_save_df()
    test_info_df = set_save_df()
    # read data
    df = pd.read_csv(config['path_setting']['metadata'])
    data_df = df.iloc[:, :-5]
    kfold_df = df.iloc[:, -5:]
    for fold_i, n_fold in enumerate(kfold_df.columns):
        print(f"Begin to test in {n_fold}")
        val = kfold_df[n_fold]
        train_df = data_df[val]
        test_df = data_df[~val]

        train_data, test_data = load_data(train_df, test_df, n_fold, config['path_setting']['pickle_path'])

        size_train = len(train_data)
        # resampler
        # train_data, val_data = random_split(train_data, [int(0.8 * size_train), size_train - int(0.8 * size_train)])
        val_indice = [i for i in range(size_train) if i % 5 == 0]
        train_indice = [i for i in range(size_train) if i % 5 != 0]

        val_data = Subset(train_data, val_indice)
        train_data = Subset(train_data, train_indice)

        print(val_data[:200]["mortality"])

        # resampler
        all_label = train_data[:]["mortality"]
        class_sample_count = np.array([len(np.where(all_label == t)[0]) for t in np.unique(all_label)])
        weight = 1. / class_sample_count
        weight = weight * [1, config['exp_setting']['class_weight']]
        # print(weight)
        samples_weight = np.array([weight[t] for t in all_label])
        samples_weight = torch.from_numpy(samples_weight).double()
        augmentation_training_size = len(samples_weight)
        sampler = WeightedRandomSampler(samples_weight, augmentation_training_size)
        # create data loader
        data_train = DataLoader(train_data, batch_size=config['exp_setting']['batch_size'], sampler=sampler)
        data_val = DataLoader(val_data, batch_size=config['exp_setting']['batch_size'])
        data_test = DataLoader(test_data, batch_size=config['exp_setting']['batch_size'])
       # set model
        # # set embedding
        bert_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        ### the grads in embedding are not fix
        pretrained_embeds = bert_model.embeddings
        pretrained_embeds.to(device)
        # ------------------------------------------------------------
        if config['exp_setting']['model_name'] == 'notes_attention':
            model = NoteSelfAttention(config['model_setting']['xai_transformer'], pretrained_embeds)
        elif config['exp_setting']['model_name'] == 'notes_rnn':
            model = NoteRNN(config['model_setting']['RNN'], pretrained_embeds)
        elif config['exp_setting']['model_name'] == 'notes_lstm':
            model = NoteLSTM(config['model_setting']['LSTM'], pretrained_embeds)
        elif config['exp_setting']['model_name'] == 'notes_rcnn':
            model = NoteRCNN(config['model_setting']['RCNN'], pretrained_embeds)
        elif config['exp_setting']['model_name'] == 'notes_gru':
            model = NoteGRU(config['model_setting']['GRU'], pretrained_embeds)
        elif config['exp_setting']['model_name'] == 'notes_attrnn':
            model = NoteAttRNN(config['model_setting']['AttRNN'], pretrained_embeds)
        # elif config['exp_setting']['model_name'] == 'notes_transformer':
        #     model = NoteTransformer(config['model_setting']['Transformer'])
        # elif config['exp_setting']['model_name'] == 'notes_bert':
        #     model = NoteBERT(config['model_setting']['BERT'])
        else:
            assert Exception("no method.")

        # unfreeze_layers = ['layer.10', 'layer.11', 'bert.pooler']
        # for name, params in model.named_parameters():
        #     if "embeddings" in name or "encoder" in name:
        #         params.requires_grad = False
        #         # for ele in unfreeze_layers:
        #         #     if ele in name:
        #         #         params.requires_grad = True
        model.to(device)

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['exp_setting']['lr'])
        scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.98)
        early_stopping = AUCEarlyStopping(patience=3)

        best_auc = 0
        total_epochs = config['exp_setting']['epochs']
        # train model
        for epoch in range(total_epochs):
            train_acc, train_loss = train_epoch(model, data_train, optimizer, device, epoch, total_epochs, augmentation_training_size)
            val_loss, auc_roc, auc_pr, acc, sensitive, specificty = eval_model(model, data_val, device)
            print(
                '[epoch {}] val_loss: {:.3f} val_auc_roc: {:.3f} val_auc_pr: {:.3f} val_acc: {:.3f} val_sen: {:.3f} val_spec: {:.3f}'.format(
                    epoch + 1, val_loss, auc_roc, auc_pr, acc, sensitive, specificty))
           # save best model
            if auc_roc > best_auc:
                torch.save(model,
                           os.path.join(config['path_setting']['ckpt_dir'], "best_model_state_{}.bin".format(fold_i)))
                best_auc = auc_roc
            # early stop
            early_stopping(auc_roc)
            # early_stopping(val_loss)
            if early_stopping.early_stop:
                break
            # learning rate decay
            scheduler.step()
            print("INFO: %d Epoch lr: %f" % (epoch + 1, optimizer.param_groups[0]['lr']))
            # save train info
            # write csv
            train_info_df.loc[fold_i * total_epochs + epoch, "fold"] = fold_i
            train_info_df.loc[fold_i * total_epochs + epoch, "epoch"] = epoch + 1
            train_info_df.loc[fold_i * total_epochs + epoch, "loss"] = val_loss
            train_info_df.loc[fold_i * total_epochs + epoch, "auc_roc"] = auc_roc
            train_info_df.loc[fold_i * total_epochs + epoch, "auc_pr"] = auc_pr
            train_info_df.loc[fold_i * total_epochs + epoch, "acc"] = acc

            if (epoch + 1) % 3 == 0:
                train_info_df.to_csv(os.path.join(config['path_setting']['ckpt_dir'], "train_info.csv"), index=False)

        best_model = torch.load(os.path.join(config['path_setting']['ckpt_dir'], "best_model_state_{}.bin".format(fold_i)))
        test_loss, auc_roc, auc_pr, acc, sensitive, specificty = eval_model(best_model, data_test, device)
        print(
            '[test set] test_loss: {:.3f} test_auc_roc: {:.3f} test_auc_pr: {:.3f} test_acc: {:.3f} test_sen: {:.3f} test_spec: {:.3f}'.format(
                test_loss, auc_roc, auc_pr, acc, sensitive, specificty))
        # write csv
        test_info_df.loc[fold_i, "fold"] = fold_i
        test_info_df.loc[fold_i, "epoch"] = "/"
        test_info_df.loc[fold_i, "loss"] = test_loss
        test_info_df.loc[fold_i, "auc_roc"] = auc_roc
        test_info_df.loc[fold_i, "auc_pr"] = auc_pr
        test_info_df.loc[fold_i, "acc"] = acc
        test_info_df.to_csv(os.path.join(config['path_setting']['ckpt_dir'], "test_info.csv"), index=False)


if __name__ == "__main__":
    args = get_args()
    config = get_config_from_json(args.config)
    config['exp_setting']['device_id'] = args.device_id

    time = str(datetime.now()).split('.')[0].replace(':', '-').replace(' ', '-')
    ckpt_dir = f"exp_model_performance/ckpt/note/{time}/"
    config['path_setting']['ckpt_dir'] = ckpt_dir

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
        # Copy config file into result dir
        shutil.copyfile(args.config, ckpt_dir + 'config.json')

    train_and_eval(config)
