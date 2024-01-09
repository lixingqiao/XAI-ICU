import argparse
import json
import random
import os

import numpy as np
import torch

from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, precision_recall_curve, auc


def set_seed(seed):
    # os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)


# def set_seed(seed):
#     # seed init.
#     random.seed(seed)
#     np.random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#
#     # torch seed init.
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.enabled = False # train speed is slower after enabling this opts.
#
#     # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
#     os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
#     os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:2'
#
#     # avoiding nondeterministic algorithms (see https://pytorch.org/docs/stable/notes/randomness.html)
#     torch.use_deterministic_algorithms(True)
#     torch.set_deterministic(True)

    # torch.cuda.set_rng_state_all(seed)

def convert_onthot(arr, num_class=2):
    # arr : [0,1,1,0]
    # out : [[1,0],[0,1],[0,1],[1,0]]
    return np.eye(num_class)[arr]


def evaluate_metrics(y_true, y_preds):
    """
    :param y_true: [1,1,1,1,1,1]
    :param y_preds: [1,1,1,1,1,1]
    :return:
    """
    y_preds_round = np.round(y_preds)
    auc_roc = roc_auc_score(y_true, y_preds, average=None)
    # calculate auc_pr
    precision, recall, _thresholds = precision_recall_curve(y_true, y_preds)
    auc_pr = auc(recall, precision)

    acc = accuracy_score(y_true, y_preds_round)
    tn, fp, fn, tp = confusion_matrix(y_true, y_preds_round).ravel()
    sensitive = tp / (tp + fn)
    specificity = tn / (fp + tn)
    return auc_roc, auc_pr, acc, sensitive, specificity


def get_args():
    """
    Parse argument
    :return: args
    """
    argparser = argparse.ArgumentParser(__doc__)
    argparser.add_argument(
        '-c', '--config',
        required=True,
        default='None',
        help='The Configuration json file')
    argparser.add_argument('-d', '--device-id', type=int, default=0)
    args = argparser.parse_args()
    return args


def get_config_from_json(json_file):
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    return config_dict


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True



class AUCEarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_auc = None
        self.early_stop = False
    def __call__(self, val_auc):
        if self.best_auc == None:
            self.best_auc = val_auc
        elif self.best_auc - val_auc < self.min_delta:
            self.best_auc = val_auc
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_auc - val_auc >= self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True