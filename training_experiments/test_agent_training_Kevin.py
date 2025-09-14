"""
Test the interactions of agents and datasets.
"""
import sys
sys.path.append('..')
from utils import goto_root_dir
from datasets import Dataset
from agents import RNNAgent
from pathlib import Path
from training_experiments import training, config_control


# this supports call by both console and main.py
# __name__ ==  '__main__' is required by multiprocessing!
if __name__ ==  '__main__' or '.' in __name__:
      ### test the whole training procedure of rnn agents; set max_epoch_num=2000 for full training
      base_config = {
            ### dataset info
            'dataset': 'MillerRat',
            'behav_format': 'tensor',
            'behav_data_spec': {'animal_name': 'm55', #'block_truncation': (0, 712)
                                },
            ### model info
            'agent_type': 'RNN',
            'rnn_type': 'GRU', # which rnn layer to use
            'input_dim': 3, # input dimension
            'hidden_dim': 1, # dimension of this rnn layer
            'output_dim': 2, # dimension of action
            'device': 'cpu', # mostly cpu
            'output_h0': True, # whether initial hidden state included in loss
            'trainable_h0': False, # the agent's initial hidden state trainable or not
            'readout_FC': True, # whether the readout layer is full connected or not
            'one_hot': False, # whether the data input is one-hot or not
            ### training info for one model
            'lr':0.005, # learning rate
            'l1_weight': 1e-5, # L1 regularization
            'weight_decay': 0, # L2 regularization
            'penalized_weight': 'rec', # weights under regularization
            'max_epoch_num': 20, #2000, # training epochs before stopping
            'early_stop_counter': 200, # epochs after overfiting happens
            ### training info for many models on dataset
            'outer_splits': 2, # k-folds for train+val dataset and test dataset separation
            'inner_splits': 2, # k-folds for train dataset and val dataset separation
            'seed_num': 1, # number of seeds in each fold
            ### additional training info
            'save_model_pass': 'minimal', # 'full' for saving all results; 'minimal' for saving only the losses; 'none' for not saving results
            'training_diagnose': None, # can be a list of diagnose function strings
            ### current training exp path
            'exp_folder': config_control.get_training_exp_folder_name(__file__), # folder name of current training exp
      }

      config_ranges = { # keys are also used to generate model names
            'rnn_type': ['GRU'], # which rnn layer to use
            'hidden_dim': [1], # dimension of this rnn layer
            'readout_FC': [True], # whether the readout layer is full connected or not
            'l1_weight': [1e-5, 1e-4, 1e-3],
      }
      configs = config_control.vary_config(base_config, config_ranges, mode='combinatorial') # generate all configs

      for c in configs:
            training.behavior_cv_training(c, n_jobs=1, verbose=True)