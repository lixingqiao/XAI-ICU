import os

import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm


def process(args):
    notes_cnt = 0
    total_notes = 0
    data_dir = args.data_path
    notes = read_notes_table(args.mimic3_path)

    for subject_dir in tqdm(os.listdir(data_dir), desc='Iterating over subjects'):
        # skip meta data
        dn = os.path.join(data_dir, subject_dir)
        if not os.path.isdir(dn):
            continue

        subject_id = int(subject_dir)
        subject_notes = notes.loc[notes.SUBJECT_ID == subject_id]
        if subject_notes.shape[0] == 0:
            continue
        stays = read_stays(dn)
        merged_df = subject_notes.merge(stays, left_on=['HADM_ID'], right_on=['HADM_ID'],
                                        how='left', suffixes=['', '_r'], indicator=True)
        assert len(merged_df) == len(subject_notes)
        merged_df = merged_df[merged_df['_merge'] == 'both']
        merged_df = merged_df[['ICUSTAY_ID', 'CHARTTIME', 'TEXT', 'CATEGORY']]
        for i in range(stays.shape[0]):
            stay_id = stays.ICUSTAY_ID.iloc[i]
            intime = stays.INTIME.iloc[i]
            outtime = stays.OUTTIME.iloc[i]

            notes_episode = get_events_for_stay(merged_df, stay_id, intime, outtime)

            if notes_episode.shape[0] == 0:
                # no notes for this episode
                continue

            notes_episode = add_hours_elpased_to_events(notes_episode, intime)

            columns = list(notes_episode.columns)
            columns_sorted = sorted(columns, key=(lambda x: "" if x == "HOURS" else x))

            notes_episode = notes_episode[columns_sorted].sort_values(by="HOURS")
            notes_episode.to_csv(os.path.join(dn, "notes{}_timeseries.csv".format(i + 1)), index=False)
            total_notes += notes_episode.shape[0]
            notes_cnt += 1
    print(f"ICU stays with notes {notes_cnt}\n" +
          f"Total notes {total_notes}")


def add_hours_elpased_to_events(events, dt, remove_charttime=True):
    events = events.copy()
    events['HOURS'] = (events.CHARTTIME - dt).apply(lambda s: s / np.timedelta64(1, 's')) / 60. / 60
    if remove_charttime:
        del events['CHARTTIME']
    return events


def get_events_for_stay(notes, icustayid, intime=None, outtime=None):
    # icustay_id is right or charttime is in icu charge time
    idx = (notes.ICUSTAY_ID == icustayid)
    # if intime is not show hours / minutes / second, this code will make wrong notes record.
    # sometimes the notes.csv hadm_id is wrong, so the icustay_id is also wrong, in this case, notes will be extracted by in/out time
    if intime is not None and outtime is not None:
        idx = idx | ((notes.CHARTTIME >= intime) & (notes.CHARTTIME <= outtime))
    notes = notes[idx]
    return notes


def read_notes_table(mimic3_path):
    notes = pd.read_csv(os.path.join(mimic3_path, 'NOTEEVENTS.csv'),
                        header=0, index_col=0, dtype={'HADM_ID': str, 'SUBJECT_ID': int})
    notes.CHARTTIME = pd.to_datetime(notes.CHARTTIME)
    # remove notes without charttime
    notes = notes.loc[~notes.CHARTTIME.isnull()]
    # remove notes with error
    notes = notes.loc[notes.ISERROR.isnull()]
    # remove notes without hadm_id
    notes = notes.loc[~notes.HADM_ID.isnull()]
    # select specified type (in fact, this is the top three most categories in tables)
    # notes = notes.loc[notes.CATEGORY.isin(['Nursing', 'Nursing/other', 'Physician '])]
    return notes


def read_stays(subject_path):
    stays = pd.read_csv(os.path.join(subject_path, 'stays.csv'),
                        header=0, index_col=None, dtype={'HADM_ID': str})
    stays.INTIME = pd.to_datetime(stays.INTIME)
    stays.OUTTIME = pd.to_datetime(stays.OUTTIME)
    stays.DOB = pd.to_datetime(stays.DOB)
    stays.DOD = pd.to_datetime(stays.DOD)
    stays.DEATHTIME = pd.to_datetime(stays.DEATHTIME)
    stays.sort_values(by=['INTIME', 'OUTTIME'], inplace=True)
    return stays


def main():
    parser = argparse.ArgumentParser(description='Extract notes for per-subject data.')
    parser.add_argument('mimic3_path', type=str, help='Directory containing MIMIC-III CSV files.')
    parser.add_argument('data_path', type=str, help='Directory containing subject sub-directories.')
    args, _ = parser.parse_known_args()

    process(args)


if __name__ == '__main__':
    main()
