"""
Test the interactions of agents and datasets.
"""
import sys
sys.path.append('..')
from utils import goto_root_dir
from pathlib import Path
from training_experiments.training import *


# this supports call by both console and main.py
# __name__ ==  '__main__' is required by multiprocessing!
if __name__ ==  '__main__' or '.' in __name__:
      ### test the whole training procedure of cog agents
      base_config = {
            ### dataset info
            'dataset': 'BartoloMonkey',
            'behav_format': 'cog_session',
            'behav_data_spec': {'animal_name': 'V', 'filter_block_type': 'both', 'block_truncation': (10, 70)}, # 'both' for all blocks
            ### model info
            'agent_type': 'PRLCog',
            'cog_type': 'MB0',
            'seed': 0,
            ### training info for one model
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

      config_ranges = { # keys are also used to generate model names
            'cog_type': ['MB0'],
      }
      behavior_cv_training_config_combination(base_config, config_ranges, n_jobs=-1, verbose_level=1)