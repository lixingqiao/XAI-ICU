from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
import re
import multiprocessing

import yaml

import numpy as np
import pandas as pd
import wfdb

# ABP sys 收缩压 ABP dias 舒张压
# NBP sys 收缩压 NPB dias 舒张压 （非侵入）
# SpO2 血样饱和度 PLUSE 脉搏 RESP 呼吸率 HR 心率
# PVC_min 早期心室收缩每分钟次数 （正常小于 5次/min）
# CVP 中心静脉压 （5～12）

STANDARD_SIGNAL = {
    "PULSE": "PULSE",
    "RESP": "RESP",
    "HR": "HR",
    "CVP": "CVP",
    'Atrial Paced Beat Count': 'Atrial_paced_beat_count',
    'Normal Beat Count': 'Normal_beat_count',
    'All Beat Count': 'All_beat_count',
    'pNN50 Percent': 'pNN50_percent',
    'Bigeminy Percent': 'Bigeminy_percent',
    'Vent Paced Percent': 'Vent_paced_percent'
}

for i in ["PVC Rate per Minute", "PVC Count"]:
    STANDARD_SIGNAL[i] = "PVC_count"

for i in ["%SpO2", "SpO2"]:
    STANDARD_SIGNAL[i] = "SpO2"
for i in ["ABP SYS", "ABP Sys", "ABPSys"]:
    STANDARD_SIGNAL[i] = "ABP_sys"
for i in ["ABP DIAS", "ABP Dias", "ABPDias"]:
    STANDARD_SIGNAL[i] = "ABP_dias"
for i in ["ABP MEAN", "ABP Mean", "ABPMean"]:
    STANDARD_SIGNAL[i] = "ABP_mean"
for i in ["NBP SYS", "NBP Sys", "NBPSys"]:
    STANDARD_SIGNAL[i] = "NBP_sys"
for i in ["NBP DIAS", "NBP Dias", "NBPDias"]:
    STANDARD_SIGNAL[i] = "NBP_dias"
for i in ["NBP MEAN", "NBP Mean", "NBPMean"]:
    STANDARD_SIGNAL[i] = "NBP_mean"
for i in ["PAP MEAN", "PAP Mean", "PAPMean"]:
    STANDARD_SIGNAL[i] = "PAP_mean"
for i in ["PAP SYS", "PAP Sys", "PAPSys"]:
    STANDARD_SIGNAL[i] = "PAP_sys"
for i in ["PAP DIAS", "PAP Dias", "PAPDias"]:
    STANDARD_SIGNAL[i] = "PAP_dias"

SIGNAL_NAME = list(set(STANDARD_SIGNAL.values()))

REMOVE_WORDS = [
    'renal failure',
    'cerebrovascular disease',
    'myocardial infarction',
    'Cardiac dysrhythmias',
    'kidney disease',
    'pulmonary disease',
    'bronchiectasis',
    'surgical procedures',
    'Conduction disorders',
    'Congestive heart failure',
    'nonhypertensive',
    'Coronary atherosclerosis',
    'heart disease',
    'Diabetes mellitus',
    'lipid metabolism',
    'Essential hypertension',
    'Fluid disorders',
    'electrolyte disorders',
    'Gastrointestinal hemorrhage',
    'secondary hypertension',
    'liver diseases',
    'respiratory disease',
    'Pleurisy',
    'pneumothorax',
    'pulmonary collapse',
    'Pneumonia',
    'Respiratory failure',
    'Septicemia',
    'Shock']


def read_stays(subject_path):
    stays = pd.read_csv(os.path.join(subject_path, 'stays.csv'),
                        header=0, index_col=False, dtype={'HADM_ID': str})
    stays.INTIME = pd.to_datetime(stays.INTIME)
    stays.OUTTIME = pd.to_datetime(stays.OUTTIME)
    stays.DOB = pd.to_datetime(stays.DOB)
    stays.DOD = pd.to_datetime(stays.DOD)
    stays.DEATHTIME = pd.to_datetime(stays.DEATHTIME)
    stays.sort_values(by=['INTIME', 'OUTTIME'], inplace=True)
    return stays


def clean_note(note):
    note = note.replace('\n', ' ')
    note = note.replace('\r', ' ')
    note = note.replace('\t', ' ')
    note = note.replace('w/', 'with')
    note = note.replace('#', ' ')
    note = re.sub('[0-9]+\.', '', note)  # remove '1.', '2.'
    note = re.sub('(-){2,}|_{2,}|={2,}', '', note)  # remove __
    note = re.sub('dr\.', 'doctor', note)  # replace doctor
    note = re.sub('m\.d\.', 'md', note)  # replace md
    note = re.sub(r"(1[0-2]|[1-9]):[0-5][0-9] *(a|p|A|P)(m|M) *", ' ', note)  # remove time
    # remove phenotype label
    for words in REMOVE_WORDS:
        note = re.sub(words, ' ', note, flags=re.I)
    # replace [**patterns**] with spaces.
    note = re.sub(r'\[\*\*.*?\*\*\]', ' ', note)
    return note


def preprocess_label(label_file):
    label_dict = {}
    label_df = pd.read_csv(label_file)
    # empty label file
    if label_df.shape[0] == 0:
        return None
    assert label_df.shape[0] == 1
    label_dict["mortality"] = int(label_df.iloc[0]["Mortality"])
    label_dict["los"] = 24.0 * label_df.iloc[0]['Length of Stay']  # in hours
    if pd.isnull(label_dict["los"]):
        return None
    return label_dict


def preprocess_diagnose(diagnose_file, icustay, id_to_group, code_to_group, group_to_id, definitions):
    cur_labels = [0 for i in range(len(id_to_group))]
    diagnoses_df = pd.read_csv(diagnose_file, dtype={"ICD9_CODE": str})
    diagnoses_df = diagnoses_df.loc[diagnoses_df.ICUSTAY_ID == icustay]
    # each icu stays must have the corresponed diagnoses.
    assert diagnoses_df.shape[0] > 0
    for index, row in diagnoses_df.iterrows():
        if row['USE_IN_BENCHMARK']:
            code = row['ICD9_CODE']
            group = code_to_group[code]
            group_id = group_to_id[group]
            cur_labels[group_id] = 1
    cur_labels = [x for (i, x) in enumerate(cur_labels)
                  if definitions[id_to_group[i]]['use_in_benchmark']]
    return cur_labels


def preprocess_physi(physi_file, out_file, min_los=24.0, eps=1e-6):
    if not os.path.exists(physi_file):
        return None
    ts_df = pd.read_csv(physi_file, index_col=False, header=0)
    ts_df = ts_df.loc[(ts_df.Hours > -eps) & (ts_df.Hours < min_los + eps)]
    if ts_df.shape[0] == 0:
        return None
    ts_df.to_csv(out_file, index=False)
    return out_file


def preprocess_note(note_file, out_file, min_los=24.0, eps=1e-6):
    if not os.path.exists(note_file):
        return None
    note_df = pd.read_csv(note_file, index_col=False, header=0)
    note_df = note_df.loc[(note_df.HOURS > -eps) & (note_df.HOURS < min_los + eps)]
    if note_df.shape[0] == 0:
        return None
    note_df = note_df.drop_duplicates(subset=["HOURS"], keep="last", inplace=False)
    note_df["TEXT"] = note_df["TEXT"].apply(clean_note)
    note_df = note_df.drop_duplicates(subset=["TEXT"], keep="last", inplace=False)
    note_df.to_csv(out_file, index=False)
    return out_file


def preprocess_wdb(wdb_file, out_file, wdb_path="../physionet.org/files/mimic3wdb-matched/1.0/", select_hours=3.0,
                   min_los=24.0):
    if not os.path.exists(wdb_file):
        return None
    wdb_df = pd.read_csv(wdb_file, index_col=False, header=0)
    during_time = wdb_df.iloc[0]["DURATION"]

    assert wdb_df.shape[0] == 1
    assert during_time + min(0, wdb_df.iloc[0]["HOURS"]) >= select_hours
    assert min_los - select_hours >= wdb_df.iloc[0]["HOURS"]

    wdb_data = wfdb.rdrecord(os.path.join(wdb_path, wdb_df.iloc[0]["wdb_path"]))

    assert wdb_data.fs == 125

    if wdb_df.iloc[0]["HOURS"] < 0:
        start_signal = int(abs(wdb_df.iloc[0]["HOURS"]) * 60 * 60 * wdb_data.fs)
    else:
        start_signal = 0

    select_signal_num = int(select_hours * 60 * 60 * wdb_data.fs)

    if "II" not in wdb_data.sig_name:
        return None
    ECG_v2_index = wdb_data.sig_name.index("II")
    ECG_v2 = wdb_data.p_signal[start_signal:start_signal + select_signal_num, ECG_v2_index]
    ECG_v2 = np.nan_to_num(ECG_v2, nan=0.0)
    # ECG_v2 = ECG_v2[~np.isnan(ECG_v2)]
    # if len(ECG_v2_dropna) < select_signal_num:
    #     return None
    # ECG_v2_select = ECG_v2_dropna[:select_signal_num]
    ECG_v2_select = ECG_v2.astype(np.float32)
    np.save(out_file, ECG_v2_select)
    return out_file


def preprocess_vital_sign(vital_sign_file, out_file, vital_sign_path="../physionet.org/files/mimic3wdb-matched/1.0/",
                          select_hours=12, min_los=24.0, eps=1e-6, fs=0.016666):
    if not os.path.exists(vital_sign_file):
        return None
    vital_sign_df = pd.read_csv(vital_sign_file, index_col=False, header=0)
    during_time = vital_sign_df.iloc[0]["DURATION"]

    assert vital_sign_df.shape[0] == 1
    assert during_time + min(0, vital_sign_df.iloc[0]["HOURS"]) >= select_hours
    assert min_los - select_hours >= vital_sign_df.iloc[0]["HOURS"]

    vital_sign_data = wfdb.rdrecord(os.path.join(vital_sign_path, vital_sign_df.iloc[0]["vital_path"]))

    sig_name = list(map(lambda x: STANDARD_SIGNAL.get(x, x), vital_sign_data.sig_name))
    # assert vital_sign_data.p_signal.shape[1] == len(sig_name)
    data_df = pd.DataFrame(vital_sign_data.p_signal, columns=sig_name)

    data_df = data_df[[each_name for each_name in sig_name if each_name in SIGNAL_NAME]]
    # if data_df.shape[1] < 4:
    #     return None

    for each_name in SIGNAL_NAME:
        if each_name not in data_df.columns:
            data_df[each_name] = np.nan
    # remove abnormal value
    # - remove negative value
    # - substitute zero with nan
    data_df[data_df < 0] = 0
    data_df = data_df.replace(to_replace=0, value=np.nan)

    data_df["HOURS"] = [vital_sign_df.iloc[0]["HOURS"] + i * round(1 / vital_sign_data.fs) / 3600 for i in
                        range(len(data_df))]
    data_df = data_df.loc[(data_df.HOURS > -eps) & (data_df.HOURS < min_los + eps)]

    assert data_df.iloc[-1]["HOURS"] - data_df.iloc[0]["HOURS"] >= 12

    columns = list(data_df.columns)
    columns_sorted = sorted(columns, key=(lambda x: "" if x == "HOURS" else x))
    data_df = data_df[columns_sorted]
    # resample the frequency
    data_df = data_df.loc[data_df.index % int(vital_sign_data.fs / fs) == 0]

    data_df.to_csv(out_file, index=False)
    return out_file


def process(args, patients, definitions, code_to_group, id_to_group, group_to_id, p_id, min_los=24.0):
    codes_in_benchmark = [x for x in id_to_group
                          if definitions[x]['use_in_benchmark']]
    out_dict = {}
    for patient in patients:
        patient_folder = os.path.join(args.root_path, patient)
        stays = read_stays(patient_folder)
        for i in range(stays.shape[0]):
            # preprocess label
            label_file = os.path.join(patient_folder, f"episode{i + 1}.csv")
            if not os.path.exists(label_file):
                # no data for this icustay records.
                continue
            label_dict = preprocess_label(label_file)
            # empty label file (because this icu stays don't have diagnoses or los.)
            if not label_dict:
                continue
            # select los
            if label_dict["los"] < min_los:
                continue
            # set label
            out_dict.setdefault("mortality", []).append(label_dict["mortality"])
            out_dict.setdefault("los", []).append(label_dict["los"])

            # preprocess diagnose (each icu stays must have the corresponed diagnoses.)
            icustay_id = stays["ICUSTAY_ID"].iloc[i]
            diagnose_file = os.path.join(patient_folder, "diagnoses.csv")
            phenotype = preprocess_diagnose(diagnose_file, icustay_id, id_to_group, code_to_group, group_to_id,
                                            definitions)
            for diagnose_name, target in zip(codes_in_benchmark, phenotype):
                out_dict.setdefault(diagnose_name, []).append(target)

            # preprocess physi
            physi_file = os.path.join(patient_folder, f"episode{i + 1}_timeseries.csv")
            physi_out_file = os.path.join(args.output_path, "physi", patient + "_" + f"episode{i + 1}_timeseries.csv")
            physi_out = preprocess_physi(physi_file, physi_out_file, min_los=min_los)
            out_dict.setdefault("physi", []).append(physi_out)

            # preprocess note
            note_file = os.path.join(patient_folder, f"notes{i + 1}_timeseries.csv")
            note_out_file = os.path.join(args.output_path, "notes", patient + "_" + f"note{i + 1}_timeseries.csv")
            note_out = preprocess_note(note_file, note_out_file, min_los=min_los)
            out_dict.setdefault("notes", []).append(note_out)

            # preprocess wdb
            # wdb_file = os.path.join(patient_folder, f"wdb{i + 1}_timeseries.csv")
            # wdb_out_file = os.path.join(args.output_path, "wdb", patient + "_" + f"wdb{i + 1}_timeseries.npy")
            # wdb_out = preprocess_wdb(wdb_file, wdb_out_file, min_los=min_los)
            # out_dict.setdefault("wdb", []).append(wdb_out)

            # preprocess vital sign
            vital_sign_file = os.path.join(patient_folder, f"vital{i + 1}_timeseries.csv")
            vital_out_file = os.path.join(args.output_path, "vital", patient + "_" + f"vital{i + 1}_timeseries.csv")
            vital_out = preprocess_vital_sign(vital_sign_file, vital_out_file, min_los=min_los)
            out_dict.setdefault("vital", []).append(vital_out)

            # save basic info for test.
            out_dict.setdefault("subject_id", []).append(stays["SUBJECT_ID"].iloc[i])
            out_dict.setdefault("icustay_id", []).append(stays["ICUSTAY_ID"].iloc[i])
    total_len = len(out_dict["subject_id"])
    for k, v in out_dict.items():
        assert len(v) == total_len
    print(f"Finish {p_id}.")
    return out_dict


def multiprocess(args, definitions, code_to_group, id_to_group, group_to_id, n=24):
    patients = list(filter(str.isdigit, os.listdir(os.path.join(args.root_path))))
    pool = multiprocessing.Pool(processes=n)
    group_len = len(patients) // n
    result = []
    for i in range(0, len(patients), group_len):
        patient = patients[i:i + group_len]
        result.append(
            pool.apply_async(process, args=[args, patient, definitions, code_to_group, id_to_group, group_to_id, i]))
        print(f"process {i} start.")
    pool.close()
    pool.join()
    out_dict = {}
    for each_out_dict in result:
        for k, v in each_out_dict.get().items():
            out_dict.setdefault(k, []).extend(v)
    meta_df = pd.DataFrame(out_dict)
    meta_df.to_csv(os.path.join(args.output_path, "dataset.csv"), index=False)
    print("All finished!")


def main():
    parser = argparse.ArgumentParser(description="Create data for all task.")
    parser.add_argument('data_path', type=str, help="Path to root folder containing data sets.")
    parser.add_argument('output_path', type=str, help="Directory where the created data should be stored.")
    parser.add_argument('--phenotype_definitions', '-p', type=str, default=os.path.join(os.path.dirname(__file__),
                                                                                        '../resources/hcup_ccs_2015_definitions.yaml'),
                        help='YAML file with phenotype definitions.')
    args, _ = parser.parse_known_args()

    with open(args.phenotype_definitions) as definitions_file:
        definitions = yaml.safe_load(definitions_file)

    code_to_group = {}
    for group in definitions:
        codes = definitions[group]['codes']
        for code in codes:
            if code not in code_to_group:
                code_to_group[code] = group
            else:
                assert code_to_group[code] == group

    id_to_group = sorted(definitions.keys())
    group_to_id = dict((x, i) for (i, x) in enumerate(id_to_group))

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    if not os.path.exists(os.path.join(os.path.join(args.output_path, "physi"))):
        os.makedirs(os.path.join(args.output_path, "physi"))
    if not os.path.exists(os.path.join(os.path.join(args.output_path, "notes"))):
        os.makedirs(os.path.join(args.output_path, "notes"))
    if not os.path.exists(os.path.join(os.path.join(args.output_path, "wdb"))):
        os.makedirs(os.path.join(args.output_path, "wdb"))
    if not os.path.exists(os.path.join(os.path.join(args.output_path, "vital"))):
        os.makedirs(os.path.join(args.output_path, "vital"))

    # process(args, definitions, code_to_group, id_to_group, group_to_id)
    multiprocess(args, definitions, code_to_group, id_to_group, group_to_id)


if __name__ == '__main__':
    main()
