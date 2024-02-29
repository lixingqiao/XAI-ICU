import joblib
import os
import sys

# Add file and basedir to Path
basedir = os.path.split(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))[0]
sys.path.append(basedir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from utils import get_args, get_config_from_json
from train_eval_physi import train_and_eval


def parse_config(model_name, trial_params, config):
    # training setting
    config['exp_setting']['lr'] = trial_params['lr']

    batch_size = trial_params['batch_size']
    config['exp_setting']['batch_size'] = 2 ** batch_size

    config['exp_setting']['class_weight'] = trial_params['class_weight']
    config['exp_setting']['model_name'] = model_name

    # model setting
    if model_name == 'physi_attention':
        config['model_setting']['xai_transformer']['n_blocks'] = trial_params['n_blocks']
        config['model_setting']['xai_transformer']['drop_out'] = trial_params['drop_out']
        config['model_setting']['xai_transformer']['num_attention_heads'] = trial_params['num_attention_heads']
        config['model_setting']['xai_transformer']['attention_head_size'] = int(
            config['model_setting']['xai_transformer']['all_head_size'] /
            config['model_setting']['xai_transformer']['num_attention_heads'])

    elif config['exp_setting']['model_name'] == 'physi_transformer':
        config['model_setting']['Transformer']['n_blocks'] = trial_params['n_blocks']
        config['model_setting']['Transformer']['drop_out'] = trial_params['drop_out']
        config['model_setting']['Transformer']['num_attention_heads'] = trial_params['num_attention_heads']

        d_ffn = trial_params['d_ffn']
        config['model_setting']['Transformer']['d_ffn'] = d_ffn * config['model_setting']['Transformer']['hidden_size']

        config['model_setting']['Transformer']['activation'] = trial_params['activation']

    elif config['exp_setting']['model_name'] == 'physi_rnn':
        config['model_setting']['RNN']['n_blocks'] = trial_params['n_blocks']
        config['model_setting']['RNN']['drop_out'] = trial_params['drop_out']
        config['model_setting']['RNN']['nonlinearity'] = trial_params['nonlinearity']

        hidden_size = trial_params['hidden_size']
        config['model_setting']['RNN']['hidden_size'] = 2 ** hidden_size

    elif config['exp_setting']['model_name'] == 'physi_lstm':
        config['model_setting']['LSTM']['n_blocks'] = trial_params['n_blocks']
        config['model_setting']['LSTM']['drop_out'] = trial_params['drop_out']
        hidden_size = trial_params['hidden_size']
        config['model_setting']['LSTM']['hidden_size'] = 2 ** hidden_size

    elif config['exp_setting']['model_name'] == 'physi_gru':
        config['model_setting']['GRU']['n_blocks'] = trial_params['n_blocks']
        config['model_setting']['GRU']['drop_out'] = trial_params['drop_out']
        hidden_size = trial_params['hidden_size']
        config['model_setting']['GRU']['hidden_size'] = 2 ** hidden_size

    elif config['exp_setting']['model_name'] == 'physi_rcnn':
        config['model_setting']['RCNN']['n_blocks'] = trial_params['n_blocks']
        config['model_setting']['RCNN']['drop_out'] = trial_params['drop_out']
        hidden_size = trial_params['hidden_size']
        config['model_setting']['RCNN']['hidden_size'] = 2 ** hidden_size

    elif config['exp_setting']['model_name'] == 'physi_attrnn':
        config['model_setting']['AttRNN']['n_blocks'] = trial_params['n_blocks']
        config['model_setting']['AttRNN']['drop_out'] = trial_params['drop_out']
        hidden_size = trial_params['hidden_size']
        config['model_setting']['AttRNN']['hidden_size'] = 2 ** hidden_size

    return config


def main(study_dir, config):
    for each_study in os.listdir(study_dir):
        model_name = "physi_" + each_study.split(".")[0].split("_")[-1]
        study_path = os.path.join(study_dir, each_study)
        print(study_path)
        # select best five config
        study = joblib.load(study_path)
        study_df = study.trials_dataframe()
        study_df = study_df.loc[study_df["state"] == "COMPLETE"]
        assert len(study_df) >= 5
        study_df = study_df.sort_values(by=['value'], ascending=False)

        best_trials_number = study_df.iloc[0]['number']
        trial = study.trials[best_trials_number]

        if model_name != "physi_attention":
            continue
        # for each config, train & eval model
        for random_seed in range(0, 5):

            config['exp_setting']['random_seed'] = config['exp_setting']['random_seed'] + random_seed

            print(model_name, "random seed:", config['exp_setting']['random_seed'])

            model_config = parse_config(model_name, trial.params, config)

            config['path_setting']['ckpt_dir'] = os.path.join(f"exp_model_performance/ckpt/physi/{model_name}", f"seed_{config['exp_setting']['random_seed']}_trial/")
            if not os.path.exists(config['path_setting']['ckpt_dir']):
                os.makedirs(config['path_setting']['ckpt_dir'])

            train_and_eval(model_config)

if __name__ == "__main__":
    study_path = "exp_model_performance/study/physi"

    args = get_args()
    config = get_config_from_json(args.config)
    config['exp_setting']['device_id'] = args.device_id

    main(study_path, config)