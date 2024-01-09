import os
from tqdm import tqdm


import numpy as np
import pandas as pd


class Reader(object):
    def __init__(self, dataset_dir, listfile_df):
        self._dataset_dir = dataset_dir
        self._current_index = 0
        self._data = listfile_df.dropna(inplace=False)
        # self._data = pd.read_csv(os.path.join(dataset_dir, listfile_path))
        self._header = self._data.columns.to_list()
        # self._no_na_fields = [i for i in self._header if i not in ["physi", "notes", "wdb", "vital"]]
        self._phenotype = [i for i in self._header if i[0].isupper()]
        assert len(self._phenotype) == 25

    def get_number_of_examples(self):
        return len(self._data)

    def random_shuffle(self):
        self._data = self._data.sample(frac=1)

    def read_example(self, index):
        raise NotImplementedError()

    def read_next(self):
        to_read_index = self._current_index
        self._current_index += 1
        if self._current_index == self.get_number_of_examples():
            self._current_index = 0
        return self.read_example(to_read_index)


class MultiModalReader(Reader):
    def __init__(self, dataset_dir, listfile_df, period_length=24.0, modal=None):
        Reader.__init__(self, dataset_dir, listfile_df)
        self._data = self._data.loc[self._data["los"] >= period_length]
        self._period_length = period_length
        self.modal = modal

    @classmethod
    def _modal_procss_func(cls, modal):
        modal2function = {"notes": cls._read_notes,
                          "physi": cls._read_physi,
                          "wdb": cls._read_wdb,
                          "vital": cls._read_vital}
        return modal2function[modal]

    def read_example(self, index):
        example = {}
        if index < 0 or index >= len(self._data):
            raise ValueError("Index must be from 0 (inclusive) to number of lines (exclusive).")
        example["mortality"] = int(self._data.iloc[index]["mortality"])
        example["los"] = self._data.iloc[index]["los"]
        example["phenotype"] = self._data.iloc[index][self._phenotype].values
        example["phenotype_header"] = self._phenotype
        for each_modal in self.modal:
            modal_f_name = self._data.iloc[index][each_modal]
            preprocess_function = self._modal_procss_func(each_modal)
            out = preprocess_function(self._dataset_dir, modal_f_name)
            if each_modal == "physi":
                assert len(out) == 2
                example["physi"] = out[0]
                example["physi_header"] = out[1]
            elif each_modal == "vital":
                assert len(out) == 2
                example["vital"] = out[0]
                example["vital_header"] = out[1]
            else:
                example[each_modal] = out
        return example

    @staticmethod
    def _read_physi(dataset_dir, physi_filename):
        ret = []
        with open(os.path.join(dataset_dir, physi_filename), "r") as physifile:
            header = physifile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in physifile:
                mas = line.strip().split(',')
                ret.append(np.array(mas))
        return np.stack(ret), header

    @staticmethod
    def _read_wdb(dataset_dir, wdb_filename):
        wdb = np.load(os.path.join(dataset_dir, wdb_filename))
        return wdb

    @staticmethod
    def _read_notes(dataset_dir, note_filename):
        notefile = pd.read_csv(os.path.join(dataset_dir, note_filename))
        text_list = notefile["TEXT"].to_list()
        text = " ".join(text_list)
        return text

    @staticmethod
    def _read_vital(dataset_dir, vital_filename):
        vital_df = pd.read_csv(os.path.join(dataset_dir, vital_filename))
        assert len(vital_df) <= 24 * 60 + 1
        # if len(vital_df) > 24 * 60 + 1:
        #     vital_df = vital_df.loc[vital_df.index % 60 == 0]
        header = vital_df.columns.to_list()
        assert header[0] == "HOURS"
        return vital_df.values, header


def read_chunk(reader, chunk_size):
    data = {}
    for i in range(chunk_size):
        ret = reader.read_next()
        for k, v in ret.items():
            data.setdefault(k, []).append(v)
    # only save one header.
    data["phenotype_header"] = data["phenotype_header"][0]
    if "physi_header" in data:
        data["physi_header"] = data["physi_header"][0]
    elif "vital_header" in data:
        data["vital_header"] = data["vital_header"][0]
    return data
