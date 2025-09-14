"""
Run all models on all monkeys
"""
import sys
sys.path.append('..')
from training_experiments.training import *
from training_experiments.training_Nautilus_jobs_generation import *

base_config = {
      ### dataset info
      'dataset': 'BartoloMonkey',
      'behav_format': 'tensor',
      'behav_data_spec': {'animal_name': 'all', 'filter_block_type': 'both', 'block_truncation': (10, 70)},
      ### model info
      'agent_type': 'RNN',
      'rnn_type': 'GRU', # which rnn layer to use
      'input_dim': 3,
      'hidden_dim': 2, # dimension of this rnn layer
      'output_dim': 2, # dimension of action
      'device': 'cpu',
      'output_h0': True, # whether initial hidden state included in loss
      'trainable_h0': False, # the agent's initial hidden state trainable or not
      'readout_FC': True, # whether the readout layer is full connected or not
      'one_hot': False, # whether the data input is one-hot or not
      ### training info for one model
      'lr':0.005,
      'l1_weight': 1e-5,
      'weight_decay': 0,
      'penalized_weight': 'rec',
      'max_epoch_num': 2000,
      'early_stop_counter': 200,
      ### training info for many models on dataset
      'outer_splits': 10,
      'inner_splits': 9,
      'seed_num': 3,
      ### additional training info
      'save_model_pass': 'minimal', # 'full' for saving all results; 'minimal' for saving only the losses; 'none' for not saving results
      'training_diagnose': None, # can be a list of diagnose function strings
      ### current training exp path
      'exp_folder': get_training_exp_folder_name(__file__),
}

config_ranges = { # keys are used to generate model names
      'rnn_type': ['GRU'],
      'hidden_dim': [1,2,3,4,5,10,20],
      'readout_FC': [True],
      'l1_weight': [1e-5],
}

resource_dict = {'memory': 8, 'cpu': 16, 'gpu': 0}
behavior_cv_training_job_combination(base_config, config_ranges, resource_dict)

base_config.update({
      'input_dim': 4,
      'one_hot': True,
      'device': 'cpu',
})

config_ranges.update({
      'rnn_type': ['SGRU'],
      'hidden_dim': [1,2,3,4
                     ],
      'readout_FC': [True],
      'l1_weight': [1e-5],
})

resource_dict = {'memory': 8, 'cpu': 16, 'gpu': 0}
behavior_cv_training_job_combination(base_config, config_ranges, resource_dict)


config_ranges.update({
      'rnn_type': ['PNR1'],
      'hidden_dim': [1,2],
      'readout_FC': [True],
      'polynomial_order': [1],
})

behavior_cv_training_job_combination(base_config, config_ranges, resource_dict)

config_ranges.update({
      'hidden_dim': [2],
      'readout_FC': [True],
      'l1_weight': [1e-5],
      'symm': [True],
})
behavior_cv_training_job_combination(base_config, config_ranges, resource_dict)

base_config = {
      ### dataset info
      'dataset': 'BartoloMonkey',
      'behav_format': 'cog_session',
      'behav_data_spec': {'animal_name': 'all', 'filter_block_type': 'both', 'block_truncation': (10, 70)},
      # 'both' for all blocks
      ### model info
      'agent_type': 'PRLCog',
      'cog_type': 'MB0',
      'device': 'cpu',
      ### training info for one model
      ### training info for many models on dataset
      'outer_splits': 10,
      'inner_splits': 9,
      'seed_num': 3,
      ### additional training info
      'save_model_pass': 'minimal', # 'full' for saving all results; 'minimal' for saving only the losses; 'none' for not saving results
      'training_diagnose': None, # can be a list of diagnose function strings
      ### current training exp path
      'exp_folder': get_training_exp_folder_name(__file__),
}

config_ranges = {  # keys are also used to generate model names
      'cog_type': [
            # 'BAS',
            'RC', 'MB0', 'MB0s', 'MB0md', #'MB0m', 'MB0se', 'MB0mdnb',
            'MB1', 'LS0', 'LS1', 'Q(0)',

      ],
}
# if __name__ ==  '__main__' or '.' in __name__:
#       behavior_cv_training_config_combination(base_config, config_ranges, n_jobs=-1, verbose_level=1)