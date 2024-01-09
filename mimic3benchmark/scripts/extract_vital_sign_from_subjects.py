import os
import argparse

from datetime import datetime

from tqdm import tqdm

import pandas as pd
import numpy as np

import wfdb

# vital_hours = [3, 12]
# num = [7949, 7575]
LEAST_VITAL_HOURS = 12.0
MIN_LOS = 24.0


# LEAST_STAY_HOURS = 24.0


def process(args):
    vital_cnt = 0
    vital_records = read_records_vital(args.vital_path)
    subject_list = list(filter(str.isdigit, os.listdir(args.data_path)))

    for subjet_id in tqdm(subject_list, desc='Iterating over subjects'):
        dn = os.path.join(args.data_path, subjet_id)
        assert os.path.exists(dn)
        assert len(subjet_id) <= 5
        vital_id = "p" + (6 - len(subjet_id)) * "0" + subjet_id
        subject_records = vital_records.loc[vital_records["p_id"] == vital_id]
        if subject_records.shape[0] == 0:
            continue

        stays = read_stays(dn)

        for i in range(stays.shape[0]):
            stay_id = stays.ICUSTAY_ID.iloc[i]
            los = stays.LOS.iloc[i] * 24
            intime = stays.INTIME.iloc[i]
            # if length of stay less than 24 hours, drop this stay record.
            if los < MIN_LOS:
                continue
            icu_stays_records = add_hours_elpased_to_events(subject_records, intime)
            # the start (hours) of vital must begin before 24 - the least vital hours.
            icu_stays_records = icu_stays_records.loc[icu_stays_records.HOURS <= MIN_LOS - LEAST_VITAL_HOURS]
            if icu_stays_records.shape[0] == 0:
                continue
            for j in range(icu_stays_records.shape[0]):
                record_data = wfdb.rdheader(os.path.join(args.vital_path, icu_stays_records.vital_path.iloc[j]))
                duration = record_data.sig_len / (record_data.fs * 60 * 60)
                start_hours = icu_stays_records.HOURS.iloc[j]
                if duration + min(0, start_hours) >= LEAST_VITAL_HOURS:
                    vital_cnt += 1
                    save_vital_df = icu_stays_records.iloc[j:j + 1, :]
                    save_vital_df = save_vital_df.copy()
                    save_vital_df["DURATION"] = duration
                    save_vital_df["ICUSTAY_ID"] = stay_id
                    save_vital_df.to_csv(os.path.join(dn, "vital{}_timeseries.csv".format(i + 1)), index=False)
                    break
    print(f"Vital sign count: {vital_cnt}")


def read_records_vital(vital_path):
    vital_records = pd.read_csv(os.path.join(vital_path, "RECORDS-numerics"), sep="/", header=None)
    vital_records.columns = ["p_dir", "p_id", "p_name"]
    vital_records["date"] = pd.to_datetime(vital_records.p_name.apply(
        lambda x: datetime.strptime(x[8:-1], "%Y-%m-%d-%H-%M")))
    vital_records["vital_path"] = vital_records.p_dir.str[:] + "/" + \
                                  vital_records.p_id.str[:] + "/" + vital_records.p_name.str[:]
    return vital_records


def add_hours_elpased_to_events(events, dt, remove_charttime=False):
    events = events.copy()
    events['HOURS'] = (events.date - dt).apply(lambda s: s / np.timedelta64(1, 's')) / 60. / 60
    if remove_charttime:
        del events['date']
    return events


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
    parser.add_argument('vital_path', type=str, help='Directory containing MIMIC-III Waveform Matched Subset.')
    parser.add_argument('data_path', type=str, help='Directory containing subject sub-directories.')
    args, _ = parser.parse_known_args()

    process(args)


if __name__ == '__main__':
    main()
