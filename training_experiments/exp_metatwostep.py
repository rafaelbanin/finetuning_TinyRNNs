"""
Run all models on meta-RL on two-step task.
Especially the blocks are segmented.
"""
import sys
sys.path.append('..')
from training_experiments.training import *
from training_experiments.training_Nautilus_jobs_generation import *

for seed in [#0,
             4
             ]:
      base_config = {
            ### dataset info
            'dataset': 'MetaTwoStep',
            'behav_format': 'cog_session',
            'behav_data_spec': {'seed': seed},
            # 'both' for all blocks
            ### model info
            'agent_type': 'RTSCog',
            'cog_type': 'MB0',
            'device': 'cpu',
            ### training info for one model
            ### training info for many models on dataset
            'outer_splits': 10,
            'single_inner_fold': True,
            'inner_splits': 9,
            'seed_num': 3,
            ### additional training info
            'save_model_pass': 'minimal', # 'full' for saving all results; 'minimal' for saving only the losses; 'none' for not saving results
            'training_diagnose': None, # can be a list of diagnose function strings
            ### current training exp path
            'exp_folder': get_training_exp_folder_name(__file__)+ '_seed'+str(seed),
      }

      config_ranges = {  # keys are also used to generate model names
            'cog_type': [#'BAS',
                  'LS0',  'Q(0)','Q(1)', #'LS1',
                         'RC',
                         'MFs',
                  'MB0', 'MB0s', 'MB1', # 'MB0md', 'MB0m',
                         ],
      }
      if __name__ ==  '__main__' or '.' in __name__:
            behavior_cv_training_config_combination(base_config, config_ranges, n_jobs=-1, verbose_level=1)