"""
Run all models on monkey V trained initialy 
on syntetic data and later done an fine tuning 
on real data.

Start by running allagents_monkey_all.py in 
simulating_experiments
"""

import sys
sys.path.append('..')
from training_experiments.training import *
from training_experiments.training_Nautilus_jobs_generation import *


'''
base_config = { ########### Treino base com dados sintéticos #################
      ### dataset info
      'dataset': 'SimAgent',
      'behav_format': 'tensor',
      'behav_data_spec': ['agent_path', 'agent_name'],
      'agent_path': ['allagents_monkeyV_nblocks100_ntrials100'],
      'agent_name': 'LS0_seed0',
      ### model info
      'agent_type': 'RNN',
      'rnn_type': 'GRU', # which rnn layer to use
      'input_dim': 3,
      'hidden_dim': 2, # dimension of this rnn layer
      'output_dim': 2, # dimension of action
      'device': 'cuda',
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
      'trainval_percent': 0, # subsample the training-validation data (percentage) ######
      'outer_splits': 10,
      'inner_splits': 9,
      'seed_num': 3,
      ### additional training info
      'save_model_pass': 'full', # 'full' for saving all results; 'minimal' for saving only the losses; 'none' for not saving results
      'training_diagnose': None, # can be a list of diagnose function strings
      ### current training exp path
      'exp_folder': get_training_exp_folder_name(__file__),
      'model_based': 'sintetic',
}

trainval_percent_list = [100, 
                         90, 80, 70, 60, 
                         50, 40, 30, 20, 
                         10,
                         0
                         ]


config_ranges = { # keys are used to generate model names
      'agent_name': [#'GRU_1_seed0',
           # 'GRU_2_seed0',
            #'SGRU_1_seed0',
            #  'RC_seed0',
            # 'MB0s_seed0',
            # 'LS0_seed0',
            # 'MB0_seed0', 
            'MB1_seed0',
                     ],
      'rnn_type': ['GRU'],
      'hidden_dim': [#1,
                     2,
                     #3,4
            #50, 20,10,5
                     ],
      'model_based': ['sintetic'],
      'trainval_percent': trainval_percent_list,

}

behavior_cv_training_config_combination(base_config, config_ranges, n_jobs=-1, verbose_level=1)
resource_dict = {'memory': 5, 'cpu': 16, 'gpu': 0}
behavior_cv_training_job_combination(base_config, config_ranges, {'memory': 12, 'cpu': 1, 'gpu': 1})


'''
base_config = {
      ### dataset info ############### Treino com dados normais ##############
      'dataset': 'BartoloMonkey',
      'behav_format': 'tensor',
      'behav_data_spec': {'animal_name': 'V', 'filter_block_type': 'both', 'block_truncation': (10, 70)},
      'agent_path': ['allagents_monkeyV_nblocks100_ntrials100'],
      'agent_name': 'LS0_seed0',
      ### model info
      'agent_type': 'RNN',
      'rnn_type': 'GRU', # which rnn layer to use
      'input_dim': 3,
      'hidden_dim': 2, # dimension of this rnn layer
      'output_dim': 2, # dimension of action
      'device': 'cuda',
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
      'save_model_pass': 'full', # 'full' for saving all results; 'minimal' for saving only the losses; 'none' for not saving results
      'training_diagnose': None, # can be a list of diagnose function strings
      ### current training exp path
      'model_based': 'base',
      'exp_folder': get_training_exp_folder_name(__file__),
      
}

trainval_percent_list = [
			0
                         ]
'''
config_ranges = { # keys are used to generate model names
      'agent_name': [#'GRU_1_seed0',
           # 'GRU_2_seed0',
            #'SGRU_1_seed0',
            #  'RC_seed0',
            # 'MB0s_seed0',
            # 'LS0_seed0',
            # 'MB0_seed0', 
            #'MB1_seed0',
            'No_seed'
                     ],
      'rnn_type': ['GRU'],
      'hidden_dim': [#1,
                     2,
                     #3,4
            #50, 20,10,5
                     ],
      'model_based': ['base'],
      'trainval_percent': trainval_percent_list,

}

behavior_cv_training_config_combination(base_config, config_ranges, n_jobs=-1, verbose_level=1)
'''

config_ranges = { # keys are used to generate model names
      'agent_name': [ #'MB1_seed0', 
            'Q(0)_seed0',
                     ],
      'rnn_type': ['GRU'],
      'hidden_dim': [#1,
                     2,
                     #3,4
            #50, 20,10,5
                     ],
      'finetune': [True],
      'model_based': ['100_pre_trained',
                      '70_pre_trained','50_pre_trained','20_pre_trained',
                      #'10_pre_trained','30_pre_trained','40_pre_trained','60_pre_trained','80_pre_trained','90_pre_trained'
                      ],
      'trainval_percent': trainval_percent_list,


}
behavior_cv_training_config_combination(base_config, config_ranges, n_jobs=-1, verbose_level=1)

'''
if __name__ ==  '__main__' or '.' in __name__:
      behavior_cv_training_config_combination(base_config, config_ranges, n_jobs=-1, verbose_level=1)
'''
