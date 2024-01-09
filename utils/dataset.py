import numpy as np

from torch.utils.data import Dataset
from sklearn.preprocessing import normalize

from utils import read_chunk
# from wfdb.processing import normalize_bound
# import neurokit2 as nk
# from torch_ecg.cfg import CFG
# from torch_ecg._preprocessors import PreprocManager


class MultiModalDataset(Dataset):
    def __init__(self, modals):
        self.modals = modals
        self.mortality = None
        self.phenotype = None

    def __getitem__(self, item):
        subset = {}
        for i in self.modals:
            if i == "notes":
                subset['input_ids'] = getattr(self, i)['input_ids'][item]
                subset['attention_mask'] = getattr(self, i)['attention_mask'][item]
            else:
                subset[i] = getattr(self, i)[item]
        subset["mortality"] = getattr(self, "mortality")[item]
        subset["phenotype"] = getattr(self, "phenotype")[item]
        return subset

    def __len__(self):
        assert hasattr(self, "mortality")
        return len(self.mortality)

    @classmethod
    def notes_preprocess(cls, notes_data, tokenizer):
        tokenizer.truncation_side = "left"
        notes_data = tokenizer(notes_data, padding="max_length", add_special_tokens=True,
                               max_length=512, truncation=True, return_token_type_ids=False, return_tensors="np")
        return notes_data

    @classmethod
    def physi_process(cls, physi_data, discretizer, normalizer, ts=24.0):
        physi_data = [discretizer.transform(X, end=ts)[0] for X in physi_data]
        physi_data = [normalizer.transform(X) for X in physi_data]
        return np.array(physi_data)

    @classmethod
    def vital_process(cls, vital_data, discretizer, normalizer, ts=24.0):
        vital_data = [discretizer.transform(X, end=ts)[0] for X in vital_data]
        vital_data = [normalizer.transform(X) for X in vital_data]
        return np.array(vital_data)

    @staticmethod
    def clean_wdb(wdb_array):
        config = CFG(
            random=False,
            resample={"fs": 50},
            bandpass={"filter_type": "fir", "fs": 125},
            normalize={"method": "z-score"},
        )
        ppm = PreprocManager.from_config(config)
        sig, fs = ppm(wdb_array, 125)
        return sig

    # @classmethod
    # def wdb_process(cls, wdb_data, eps=1e-9):
    #     wdb_data = np.array(wdb_data)
    #     wdb_data = wdb_data[:, 450000:]
    #     # length = 8192*2
    #     # wdb_data = np.concatenate([wdb_data[:, :length], wdb_data[:, 225000:225000+length], wdb_data[:, 450000:450000+length],
    #     #                            wdb_data[:, 675000:675000+length], wdb_data[:, 900000:900000+length], wdb_data[:, 1125000:1125000+length],
    #     #                            wdb_data[:, -length:]], axis=1)
    #     # normalize wdb for each sample
    #     # wdb_data = wdb_data.transpose()
    #     # wdb_data = (wdb_data - wdb_data.mean(axis=0)) / (wdb_data.std(axis=0) + eps)
    #     # wdb_data = wdb_data.transpose()
    #     wdb_data = np.expand_dims(wdb_data, axis=1)
    #     wdb_data = cls.clean_wdb(wdb_data)
    #     assert wdb_data.shape[1] == 1 and len(wdb_data.shape) == 3
    #     return wdb_data

    @classmethod
    def load(cls, reader, physi_discretizer=None, vital_discretizer=None, physi_normalizer=None, vital_normalizer=None, tokenizer=None,
             modals=None, explain_mode=False):
        if modals is None:
            raise Exception("Please select at least one modal.")
        else:
            total_sample = reader.get_number_of_examples()
            ret = read_chunk(reader, total_sample)
        dataset = cls(modals)
        for modal in dataset.modals:
            if modal == "physi":
                assert physi_discretizer and physi_normalizer
                dataset.physi = cls.physi_process(ret["physi"], physi_discretizer, physi_normalizer)
            elif modal == "vital":
                assert vital_discretizer and vital_normalizer
                dataset.vital = cls.vital_process(ret["vital"], vital_discretizer, vital_normalizer)
            elif modal == "notes":
                assert tokenizer
                dataset.notes = cls.notes_preprocess(ret["notes"], tokenizer)
            elif modal == "wdb":
                dataset.wdb = cls.wdb_process(ret["wdb"])
            else:
                raise Exception(f"The {modal} modal is not support.")
        dataset.mortality = np.array(ret["mortality"], dtype=np.int64)
        dataset.phenotype = np.array(ret["phenotype"], dtype=np.int64)
        return dataset
