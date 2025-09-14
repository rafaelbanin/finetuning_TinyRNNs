"""
Test the interactions of agents and datasets.
"""
import sys
sys.path.append('..')
from training_experiments.training import *

### test the whole training procedure of rnn agents; set max_epoch_num=2000 for full training
base_config = {
      ### dataset info
      'dataset': 'BartoloMonkey',
      'behav_format': 'tensor',
      'behav_data_spec': {'animal_name': 'V', 'filter_block_type': 'both', 'block_truncation': (10, 70)}, # 'both' for all blocks
      ### model info
      'agent_type': 'RNN',
      'rnn_type': 'GRU', # which rnn layer to use
      'input_dim': 3, # input dimension
      'hidden_dim': 2, # dimension of this rnn layer
      'output_dim': 2, # dimension of action
      'device': 'cuda', # mostly cpu
      'output_h0': True, # whether initial hidden state included in loss
      'trainable_h0': False, # the agent's initial hidden state trainable or not
      'readout_FC': True, # whether the readout layer is full connected or not
      'one_hot': False, # whether the data input is one-hot or not
      ### training info for one model
      'lr':0.005, # learning rate
      'l1_weight': 1e-5, # L1 regularization
      'weight_decay': 0, # L2 regularization
      'penalized_weight': 'rec', # weights under regularization
      'max_epoch_num': 200, #2000, # training epochs before stopping
      'early_stop_counter': 200, # epochs after overfiting happens
      ### training info for many models on dataset
      'outer_splits': 2, # k-folds for train+val dataset and test dataset separation
      'inner_splits': 2, # k-folds for train dataset and val dataset separation
      'seed_num': 1, # number of seeds in each fold
      ### additional training info
      'save_model_pass': 'minimal', # 'full' for saving all results; 'minimal' for saving only the losses; 'none' for not saving results
      'training_diagnose': None, # can be a list of diagnose function strings
      ### current training exp path
      'exp_folder': get_training_exp_folder_name(__file__), # folder name of current training exp
}
base_config.update({
      'input_dim': 4,
      'one_hot': True,
      'polynomial_order': 1,
})
# #
# config_ranges = {
#       'rnn_type': ['SGRU'],
#       'hidden_dim': [1,2,],
#       'readout_FC': [True],
#       'l1_weight': [1e-5, 1e-4, 1e-3],
# }
config_ranges = {
      'rnn_type': ['PNR1'],
      'hidden_dim': [2,],
      'readout_FC': [True],
      'l1_weight': [1e-5],
      'symm': [True],
}
# config_ranges = { # keys are also used to generate model names, saved in 'model_path' key
#       'rnn_type': ['GRU'], # which rnn layer to use
#       'hidden_dim': [1,2], # dimension of this rnn layer
#       'readout_FC': [True], # whether the readout layer is full connected or not
#       'l1_weight': [1e-5, 1e-4, 1e-3],
# }
# config_ranges = { # keys are also used to generate model names, saved in 'model_path' key
#       'rnn_type': ['LR'],
#       'hidden_dim': [1,2],
#       'readout_FC': [True],
#       'l1_weight': [1e-5],
#       'rank': [1,2,3],
# }
behavior_cv_training_config_combination(base_config, config_ranges, verbose_level=1)