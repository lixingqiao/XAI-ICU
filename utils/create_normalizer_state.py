import os
import argparse
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
basedir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(basedir)

import pandas as pd
from utils import MultiModalReader, VitalDiscretizer, VitalNormalizer


def main():
    parser = argparse.ArgumentParser(description='Script for creating a normalizer state')
    parser.add_argument('--store_masks', dest='store_masks', action='store_true',
                        help='Store masks that specify observed/imputed values.')
    parser.add_argument('--no-masks', dest='store_masks', action='store_false',
                        help='Do not store that specify specifying observed/imputed values.')
    parser.add_argument('--output_dir', type=str, help='Directory where the output file will be saved.',
                        default='.')
    args = parser.parse_args()

    data_df = pd.read_csv("data/dataset/multimodal.csv")
    data_df = data_df.iloc[:, :-5]

    train_reader = MultiModalReader(dataset_dir="data/",
                                    listfile_df=data_df,
                                    period_length=24.0, modal=["vital"])
    discretizer = VitalDiscretizer(timestep=0.05,
                                   store_masks=args.store_masks,
                                   impute_strategy='previous',
                                   start_time='zero')
    discretizer_header = train_reader.read_example(0)['vital_header']
    continuous_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]
    normalizer = VitalNormalizer(fields=continuous_channels)

    n_samples = train_reader.get_number_of_examples()
    for i in range(n_samples):
        if i % 100 == 0:
            print('Processed {} / {} samples'.format(i, n_samples), end='\r')
        ret = train_reader.read_example(i)
        data, new_header = discretizer.transform(ret["vital"], end=24.0)
        normalizer.feed_data(data)
    print('\n')

    # all dashes (-) were colons(:)
    file_name = 'vital_mask_{}_{}.normalizer'.format(0.05, args.store_masks)
    file_name = os.path.join(args.output_dir, file_name)
    print('Saving the state in {} ...'.format(file_name))
    normalizer.save_params(file_name)


if __name__ == "__main__":
    main()
