import pickle
import sys
import os
from datetime import datetime
import logging

import numpy as np
import pandas as pd
import warnings

from torch.optim import lr_scheduler
from tqdm import tqdm
import transformers

transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# Add file and basedir to Path
basedir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(basedir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split, Subset

from utils import set_seed, get_config_from_json, evaluate_metrics, get_args, EarlyStopping, AUCEarlyStopping
from utils import MultiModalReader, MultiModalDataset, VitalDiscretizer, VitalNormalizer, PhysiDiscretizer, PhysiNormalizer

from model import AllModel, PhysiNotesModel, PhysiVitalModel, NotesVitalModel, SubmodalPhysi, SubmodalNotes, SubmodalVital
from transformers import AutoTokenizer, AutoModel

import optuna

from optuna.trial import TrialState
from optuna.samplers import TPESampler
import joblib


def train_epoch(model, data_loader, optimizer, device, epoch, num_epochs, n_examples, modals, verbose=True):
    model = model.train()
    current_loss = 0.0
    correct_predictions = 0
    losses = []
    if verbose:
        train_bar = tqdm(data_loader)
    else:
        train_bar = data_loader
    for itr, sample in enumerate(train_bar):
        total = len(data_loader)

        input_data = {"physi": None, "notes": None, "vital": None}

        for each_modal in modals:
            if each_modal == "physi":
                input_data['physi'] = sample["physi"].type(torch.float).to(device)
            if each_modal == "notes":
                input_data['notes'] = sample["input_ids"].type(torch.long).to(device)
            if each_modal == "vital":
                input_data['vital'] = sample["vital"].type(torch.float).to(device)

        targets = sample["mortality"].type(torch.long).to(device)

        outputs = model(notes=input_data['notes'], physi=input_data['physi'], vital=input_data["vital"])

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
        if verbose:
            train_bar.desc = 'Epoch[{}/{}] | Batch[{}/{}], loss: {:.3f}'.format(epoch + 1, num_epochs, itr + 1,
                                                                                total,
                                                                                current_loss / (itr + 1))
    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, device, modals):
    model = model.eval()

    losses = []
    # correct_predictions = 0
    with torch.no_grad():
        # -----
        y_label_flag = 0
        y_score_flag = 0
        # -----
        for itr, sample in enumerate(data_loader):

            input_data = {"physi": None, "notes": None, "vital": None}

            for each_modal in modals:
                if each_modal == "physi":
                    input_data['physi'] = sample["physi"].type(torch.float).to(device)
                if each_modal == "notes":
                    input_data['notes'] = sample["input_ids"].type(torch.long).to(device)
                if each_modal == "vital":
                    input_data['vital'] = sample["vital"].type(torch.float).to(device)

            targets = sample["mortality"].type(torch.long).to(device)

            outputs = model(notes=input_data['notes'], physi=input_data['physi'], vital=input_data["vital"])

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
    df.insert(4, 'acc_pr', '')
    df.insert(5, 'acc', '')
    return df


def load_data(train_df, test_df, n_fold, pickle_path, modal=["notes", "physi", "vital"], verbose=True):
    pickle_file = os.path.join(pickle_path, n_fold + ".dat")

    if os.path.exists(pickle_file):
        if verbose:
            print("Data pickle already exist...")
        with open(pickle_file, "rb") as f:
            data_dict = pickle.load(f)
    else:
        if verbose:
            print("Generate pickles...")
        if not os.path.exists(pickle_path):
            os.makedirs(pickle_path)
        # Read data to dict
        train_reader = MultiModalReader(dataset_dir="data/",
                                        listfile_df=train_df,
                                        period_length=24.0, modal=modal)
        test_reader = MultiModalReader(dataset_dir="data/",
                                       listfile_df=test_df,
                                       period_length=24.0, modal=modal)
        # Impute the input, missing value will be filled with previous value or normal value.
        physi_discretizer = PhysiDiscretizer(timestep=1.0,
                                             store_masks=True,
                                             impute_strategy='previous',
                                             start_time='zero')

        physi_discretizer_header = physi_discretizer.transform(train_reader.read_example(0)["physi"])[1].split(',')
        physi_cont_channels = [i for (i, x) in enumerate(physi_discretizer_header) if x.find("->") == -1]
        physi_normalizer = PhysiNormalizer(fields=physi_cont_channels)
        physi_normalizer.load_params('utils/ihm_ts1.0.input_str:previous.start_time:zero.normalizer')
        # Vital discretizer and normalize the feature.
        vital_discretizer = VitalDiscretizer(timestep=0.05,
                                             store_masks=False,
                                             impute_strategy='previous',
                                             start_time='zero')

        vital_discretizer_header = vital_discretizer.transform(train_reader.read_example(0)["vital"])[1].split(',')
        vital_cont_channels = [i for (i, x) in enumerate(vital_discretizer_header) if x.find("->") == -1]
        vital_normalizer = VitalNormalizer(fields=vital_cont_channels)
        vital_normalizer.load_params("utils/vital_mask_0.05_False.normalizer")
        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        # Load data
        train_dataset = MultiModalDataset.load(train_reader, vital_discretizer=vital_discretizer,
                                               vital_normalizer=vital_normalizer,
                                               tokenizer=tokenizer,
                                               physi_discretizer=physi_discretizer, physi_normalizer=physi_normalizer,
                                               modals=modal)
        test_datset = MultiModalDataset.load(test_reader, vital_discretizer=vital_discretizer,
                                             vital_normalizer=vital_normalizer,
                                             tokenizer=tokenizer,
                                             physi_discretizer=physi_discretizer, physi_normalizer=physi_normalizer,
                                             modals=modal)

        data_dict = {"train": train_dataset, "test": test_datset}
        with open(pickle_file, "wb") as f:
            pickle.dump(data_dict, f)

    return data_dict['train'], data_dict['test']


class Objective:
    def __init__(self, params):
        self.params = params

    def __call__(self, trial):
        config = self.params
        # Set search space
        # config['exp_setting']['lr'] = trial.suggest_loguniform('lr', 1e-5, 1e-3)
        # batch_size = trial.suggest_int('batch_size', 4, 5)
        # config['exp_setting']['batch_size'] = 2 ** batch_size
        config['exp_setting']['class_weight'] = trial.suggest_float('class_weight', 1, 3, step=0.1)
        # -----------------------------------------------------------------------------------------
        if config['exp_setting']['model_name'] == 'notes_physi':
            config['model_setting']['notes_physi']["notes_hidden"] = trial.suggest_categorical("notes_hidden", [20, 50, 100, 150, 200])
        if config['exp_setting']['model_name'] == 'notes_physi_vital':
            config['model_setting']['notes_physi_vital']["notes_hidden"] = trial.suggest_categorical("notes_hidden", [20, 50, 100, 150, 200])
        # Set random seed.
        set_seed(config['exp_setting']['random_seed'])
        # set device.
        device = torch.device(f"cuda:{config['exp_setting']['device_id']}" if torch.cuda.is_available() else "cpu")
        config['exp_setting']['device'] = device
        # read data
        df = pd.read_csv(config['path_setting']['metadata'])
        data_df = df.iloc[:, :-5]
        kfold_df = df.iloc[:, -5:]
        all_loss = []
        all_auc = []
        for fold_i, n_fold in enumerate(kfold_df.columns):
            val = kfold_df[n_fold]
            train_df = data_df[val]
            test_df = data_df[~val]

            train_data, test_data = load_data(train_df, test_df, n_fold, config['path_setting']['pickle_path'], verbose=config['exp_setting']['verbose'])

            size_train = len(train_data)
            # create data loader
            # train_data, val_data = random_split(train_data,
            #                                     [int(0.8 * size_train), size_train - int(0.8 * size_train)])
            val_indice = [i for i in range(size_train) if i % 5 == 0]
            train_indice = [i for i in range(size_train) if i % 5 != 0]

            val_data = Subset(train_data, val_indice)
            train_data = Subset(train_data, train_indice)

            print(val_data[:200]["mortality"])

            all_label = train_data[:]["mortality"]
            class_sample_count = np.array([len(np.where(all_label == t)[0]) for t in np.unique(all_label)])
            assert class_sample_count.sum() == len(all_label)
            weight = 1. / class_sample_count
            weight = weight * [1, config['exp_setting']['class_weight']]

            samples_weight = np.array([weight[t] for t in all_label])
            samples_weight = torch.from_numpy(samples_weight).double()
            augmentation_training_size = len(samples_weight)
            sampler = WeightedRandomSampler(samples_weight, augmentation_training_size)

            data_train = DataLoader(train_data, batch_size=config['exp_setting']['batch_size'], sampler=sampler)
            data_val = DataLoader(val_data, batch_size=config['exp_setting']['batch_size'])

            # set model
            if 'physi' in config['exp_setting']['modal']:
                physi_module = SubmodalPhysi(config['model_setting']['physi'])
                # physi_weights = torch.load(os.path.join(config['path_setting']['physi_ckpt_path'], f"seed_{config['exp_setting']['random_seed']}_trial", f"best_model_state_{fold_i}.bin"))
                physi_weights = torch.load(os.path.join(config['path_setting']['physi_ckpt_path'], f"seed_1234_trial",
                                                        f"best_model_state_{fold_i}.bin"))
                physi_module.load_state_dict(physi_weights.state_dict(), strict=False)
                for name, parameters in physi_module.named_parameters():
                    parameters.requires_grad = False
                print("load physi weights")

            if 'vital' in config['exp_setting']['modal']:
                vital_module = SubmodalVital(config['model_setting']['vital'])
                vital_weights = torch.load(os.path.join(config['path_setting']['vital_ckpt_path'], f"seed_1234_trial",
                                                        f"best_model_state_{fold_i}.bin"))
                vital_module.load_state_dict(vital_weights.state_dict(), strict=False)
                for name, parameters in vital_module.named_parameters():
                    parameters.requires_grad = False
                print("load vital weights")

            if 'notes' in config['exp_setting']['modal']:
                bert_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
                ### the grads in embedding are not fix
                pretrained_embeds = bert_model.embeddings
                pretrained_embeds.to(device)

                notes_module = SubmodalNotes(config['model_setting']['notes'], pretrained_embeds)
                notes_weights = torch.load(os.path.join(config['path_setting']['notes_ckpt_path'],
                                                        f"seed_{config['exp_setting']['random_seed']}_trial",
                                                        f"best_model_state_{fold_i}.bin"))
                notes_module.load_state_dict(notes_weights.state_dict(), strict=False)
                for name, parameters in notes_module.named_parameters():
                    parameters.requires_grad = False
                print("load notes weights")
            # ------------------------------------------------------------
            if config['exp_setting']['model_name'] == 'notes_physi':
                model = PhysiNotesModel(config['model_setting']['notes_physi'], notes_module=notes_module,
                                        physi_module=physi_module)
            if config['exp_setting']['model_name'] == 'notes_vital':
                model = NotesVitalModel(config['model_setting']['notes_vital'], notes_module=notes_module,
                                        vital_module=vital_module)
            if config['exp_setting']['model_name'] == 'physi_vital':
                model = PhysiVitalModel(config['model_setting']['physi_vital'], vital_module=vital_module,
                                        physi_module=physi_module)
            if config['exp_setting']['model_name'] == 'notes_physi_vital':
                model = AllModel(config['model_setting']['notes_physi_vital'], notes_module=notes_module,
                                 vital_module=vital_module, physi_module=physi_module)
            else:
                assert Exception("no method.")
            model.to(device)
            # ------------------------------------------------------------
            # set optimizer
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                   lr=config['exp_setting']['lr'])
            scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.98)
            # early_stopping = EarlyStopping(patience=3)
            early_stopping = AUCEarlyStopping(patience=3)
            # save
            # best_loss = 10000
            best_auc = 0
            total_epochs = config['exp_setting']['epochs']
            # train model
            for epoch in range(total_epochs):
                train_acc, train_loss = train_epoch(model, data_train, optimizer, device, epoch, total_epochs,
                                                    augmentation_training_size, config['exp_setting']['modal'],
                                                    config['exp_setting']['verbose'])
                val_loss, auc_roc, auc_pr, acc, sensitive, specificty = eval_model(model, data_val, device, config['exp_setting']['modal'])
                if config['exp_setting']['verbose']:
                    print(
                        '[epoch {}] val_loss: {:.3f} val_auc_roc: {:.3f} val_auc_pr: {:.3f} val_acc: {:.3f} val_sen: {:.3f} val_spec: {:.3f}'.format(
                            epoch + 1, val_loss, auc_roc, auc_pr, acc, sensitive, specificty))
                if auc_roc > best_auc:
                    # best_loss = val_loss
                    best_auc = auc_roc
                # early stop
                early_stopping(auc_roc)
                if early_stopping.early_stop:
                    break
                # learning rate decay
                scheduler.step()
            # all_loss.append(best_loss)
            all_auc.append(best_auc)
            # prune (prune at least trainning 2 folds)
            if fold_i > 2:
                trial.report(np.mean(all_auc), fold_i)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        mean_auc = np.mean(all_auc)
        return mean_auc


def optuna_search(config):
    print(f"Begin to search params for {config['exp_setting']['model_name']}.")
    # set grid search samplers
    sampler = TPESampler(seed=1234)
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    study = optuna.create_study(directions=['maximize'], sampler=sampler, pruner=optuna.pruners.HyperbandPruner())
    # study = optuna.create_study(directions=['maximize'], sampler=sampler)
    study.optimize(Objective(config), n_trials=20)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    best_trial = study.best_trial

    print("Best trial:", best_trial)
    print("  Value: ", best_trial.value)

    print("Best Trail Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))

    # save the 'study' object into a pickle file for analysis
    if not os.path.exists(config['path_setting']['study_path']):
        os.makedirs(config['path_setting']['study_path'])
    joblib.dump(study, os.path.join(config['path_setting']['study_path'], f"study_{config['exp_setting']['model_name']}.joblib"))


if __name__ == "__main__":
    args = get_args()
    config = get_config_from_json(args.config)
    config['exp_setting']['device_id'] = args.device_id
    config["exp_setting"]["modal"] = sorted(config["exp_setting"]["modal"])

    model_name = "_".join(sorted(config["exp_setting"]["modal"]))
    config["exp_setting"]["model_name"] = model_name

    optuna_search(config)
