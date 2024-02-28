import os
import json

import numpy as np
import platform
import pickle


class VitalDiscretizer:
    def __init__(self, timestep=0.016666, start_time='zero', store_masks=False, impute_strategy='previous',
                 config_path=os.path.join(os.path.dirname(__file__), 'vital_discretizer_config.json')):

        with open(config_path) as f:
            config = json.load(f)
            self._id_to_channel = config['id_to_channel']
            self._channel_to_id = dict(zip(self._id_to_channel, range(len(self._id_to_channel))))
            self._normal_values = config['normal_values']

        self._header = ["HOURS"] + self._id_to_channel
        self._start_time = start_time
        self._store_masks = store_masks
        self._timestep = timestep
        self._impute_strategy = impute_strategy

    def transform(self, X, header=None, end=None):
        if header is None:
            header = self._header
        assert header[0] == "HOURS"
        eps = 1e-6

        N_channels = len(self._id_to_channel)
        ts = [float(row[0]) for row in X]
        for i in range(len(ts) - 1):
            assert ts[i] < ts[i + 1] + eps

        if self._start_time == 'relative':
            first_time = ts[0]
        elif self._start_time == 'zero':
            first_time = 0
        else:
            raise ValueError("start time is invalid.")
        # if end is set, all data will pad to max hours.
        if end is None:
            max_hours = max(ts) - first_time
        else:
            max_hours = end - first_time

        N_bins = int(max_hours / self._timestep)
        begin_pos = [i for i in range(N_channels)]

        # the categorical feature convert to onehot and write into data matrix.
        def write(data, bin_id, channel, value, begin_pos):
            channel_id = self._channel_to_id[channel]
            data[bin_id, begin_pos[channel_id]] = float(value)

        #
        data = np.zeros(shape=(N_bins, N_channels), dtype=float)
        mask = np.zeros(shape=(N_bins, N_channels), dtype=int)
        original_value = [["" for j in range(N_channels)] for i in range(N_bins)]
        # iter the data, last event in each bin is saved.
        for row in X:
            t = float(row[0]) - first_time
            # if t >= max_hours:
            if t >= N_bins * self._timestep:
                continue
            bin_id = int(t / self._timestep)
            assert 0 <= bin_id < N_bins

            for j in range(1, len(row)):
                if np.isnan(row[j]):
                    continue
                channel = header[j]
                channel_id = self._channel_to_id[channel]

                mask[bin_id][channel_id] = 1
                # last value will overwrite the former.
                write(data, bin_id, channel, row[j], begin_pos)
                # save the last data in each bin.
                original_value[bin_id][channel_id] = row[j]

        # impute missing values
        if self._impute_strategy not in ['zero', 'normal_value', 'previous', 'next']:
            raise ValueError("impute strategy is invalid")
        # previous methbbod, impute the bins feature with last value. If all are null, impute with normal value
        if self._impute_strategy in ['normal_value', 'previous']:
            prev_values = [[] for i in range(len(self._id_to_channel))]
            for bin_id in range(N_bins):
                for channel in self._id_to_channel:
                    channel_id = self._channel_to_id[channel]
                    if mask[bin_id][channel_id] == 1:
                        prev_values[channel_id].append(original_value[bin_id][channel_id])
                        continue
                    if self._impute_strategy == 'normal_value':
                        imputed_value = self._normal_values[channel]
                    if self._impute_strategy == 'previous':
                        if len(prev_values[channel_id]) == 0:
                            imputed_value = self._normal_values[channel]
                        else:
                            imputed_value = prev_values[channel_id][-1]
                    write(data, bin_id, channel, imputed_value, begin_pos)

        if self._impute_strategy == 'next':
            prev_values = [[] for i in range(len(self._id_to_channel))]
            for bin_id in range(N_bins - 1, -1, -1):
                for channel in self._id_to_channel:
                    channel_id = self._channel_to_id[channel]
                    if mask[bin_id][channel_id] == 1:
                        prev_values[channel_id].append(original_value[bin_id][channel_id])
                        continue
                    if len(prev_values[channel_id]) == 0:
                        imputed_value = self._normal_values[channel]
                    else:
                        imputed_value = prev_values[channel_id][-1]
                    write(data, bin_id, channel, imputed_value, begin_pos)

        if self._store_masks:
            data = np.hstack([data, mask.astype(np.float32)])

        # create new header
        new_header = []
        for channel in self._id_to_channel:
            new_header.append(channel)

        if self._store_masks:
            for i in range(len(self._id_to_channel)):
                channel = self._id_to_channel[i]
                new_header.append("mask->" + channel)

        new_header = ",".join(new_header)

        return data, new_header


class VitalNormalizer:
    """
    Normalize the contigous feature in input data channel.
    """

    def __init__(self, fields=None):
        self._means = None
        self._stds = None
        self._fields = None
        if fields is not None:
            self._fields = [col for col in fields]

        self._sum_x = None
        self._sum_sq_x = None
        self._count = 0

    def feed_data(self, x):
        x = np.array(x)
        self._count += x.shape[0]
        if self._sum_x is None:
            self._sum_x = np.sum(x, axis=0)
            self._sum_sq_x = np.sum(x ** 2, axis=0)
        else:
            self._sum_x += np.sum(x, axis=0)
            self._sum_sq_x += np.sum(x ** 2, axis=0)

    def save_params(self, save_file_path):
        eps = 1e-7
        with open(save_file_path, "wb") as save_file:
            N = self._count
            self._means = 1.0 / N * self._sum_x
            self._stds = np.sqrt(
                1.0 / (N - 1) * (self._sum_sq_x - 2.0 * self._sum_x * self._means + N * self._means ** 2))
            self._stds[self._stds < eps] = eps
            pickle.dump(obj={'means': self._means,
                             'stds': self._stds},
                        file=save_file,
                        protocol=2)

    def load_params(self, load_file_path):
        with open(load_file_path, "rb") as load_file:
            if platform.python_version()[0] == '2':
                dct = pickle.load(load_file)
            else:
                dct = pickle.load(load_file, encoding='latin1')
            self._means = dct['means']
            self._stds = dct['stds']

    # only for contiguous data, normalize will do.
    def transform(self, X):
        if self._fields is None:
            fields = range(X.shape[1])
        else:
            fields = self._fields
        ret = 1.0 * X
        for col in fields:
            ret[:, col] = (X[:, col] - self._means[col]) / self._stds[col]
        return ret
