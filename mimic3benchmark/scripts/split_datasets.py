from __future__ import absolute_import
from __future__ import print_function

import os
import shutil
import argparse
import pandas as pd
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold

seed = 1234

np.random.seed(seed)
random.seed(seed)


def split_train_test(process_df, save_path):
    kf = StratifiedKFold(n_splits=5)

    for n_fold, (train_index, test_index) in enumerate(kf.split(np.zeros(len(process_df)), process_df.mortality.values)):

        process_df.loc[train_index, f'fold_{n_fold}'] = True
        process_df.loc[test_index, f'fold_{n_fold}'] = False

        assert all(process_df.loc[process_df[f"fold_{n_fold}"] == False].index.values == test_index)
        assert all(process_df.loc[process_df[f"fold_{n_fold}"] == True].index.values == train_index)
        assert len(process_df) == len(test_index) + len(train_index)

    process_df.to_csv(save_path, index=False)



def main():
    parser = argparse.ArgumentParser(description='Make dataset and split into train and test sets.')
    parser.add_argument('data_path', type=str, help='Directory containing multi-modal data.')
    args, _ = parser.parse_known_args()

    # Load data
    df = pd.read_csv(os.path.join(args.data_path, 'dataset.csv'))
    df = df.loc[df["los"] >= 24.0]
    print("total samples:", len(df))

    # multimodal data
    multimodal_df = df.dropna(subset=["physi", "notes", "vital"])
    multimodal_align_df = multimodal_df.sample(frac=1).reset_index(drop=True)
    print("total multimodal aligned sample:", len(multimodal_align_df))
    split_train_test(multimodal_align_df, os.path.join(args.data_path, 'multimodal.csv'))

    # no-align time series
    physi_df = df.dropna(subset=['physi'])
    physi_df = physi_df.loc[~physi_df['icustay_id'].isin(multimodal_align_df['icustay_id'])]
    physi_no_align_df = physi_df.sample(frac=1).reset_index(drop=True)
    print("total no-aligned time series sample:", len(physi_no_align_df))
    split_train_test(physi_no_align_df, os.path.join(args.data_path, 'physi_no_align.csv'))

    # no-align notes
    notes_df = df.dropna(subset=['notes'])
    notes_df = notes_df.loc[~notes_df['icustay_id'].isin(multimodal_align_df['icustay_id'])]
    notes_no_align_df = notes_df.sample(frac=1).reset_index(drop=True)
    print("total no-aligned notes sample:", len(notes_no_align_df))
    split_train_test(notes_no_align_df, os.path.join(args.data_path, 'notes_no_align.csv'))


if __name__ == '__main__':
    main()
