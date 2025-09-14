import numpy as np
from sklearn.model_selection import KFold
import os
import sys
from pathlib import Path
import joblib
# import multiprocessing as mp
import torch.multiprocessing as mp
import gc
from agents import Agent
import pandas as pd
from datasets import Dataset
from path_settings import *
import pprint
from utils.logger import PrinterLogger
from utils import goto_root_dir, set_os_path_auto
from training_experiments import config_control
import time

def get_training_exp_folder_name(file_string):
    """Automatically generate the current training_exp folder

    Args:
        file_string: __file__

    Returns:
        the file name without .py
    """
    return os.path.basename(file_string)[:-3] # remove .pys


base_config = {
      ### dataset info
      'dataset': 'BartoloMonkey',
      'behav_format': 'tensor',
      'behav_data_spec': {'animal_name': 'V', 'filter_block_type': 'both', 'block_truncation': (10, 70)},
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
      'weitraining_experiments/exp_sim_monkeyV.pyght_decay': 0,
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
      'model_based': "sintetic",
      ### current training exp path
      'exp_folder': get_training_exp_folder_name(__file__)
}

config_ranges = { # keys are used to generate model names
      'rnn_type': ['GRU'],
      'hidden_dim': [#1,
                     2,
                     #3,4,5,10,20
                     ],
      #'readout_FC': [True],
      'l1_weight': [1e-5, 1e-4, 1e-3],
}

configs = config_control.vary_config(base_config, config_ranges, mode='combinatorial')

print(configs[0])