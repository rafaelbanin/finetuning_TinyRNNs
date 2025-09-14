"""
Test the interactions of agents and datasets.
"""
import sys
import numpy as np
sys.path.append('..')
from utils import goto_root_dir
from datasets import Dataset
from agents import Agent
from pathlib import Path
from training_experiments import training
from path_settings import *
from pathlib import Path

# this supports call by both console and main.py
# __name__ ==  '__main__' is required by multiprocessing!
if __name__ ==  '__main__' or '.' in __name__:
      goto_root_dir.run()
      config = {
            'dataset': 'BartoloMonkey',
            'behav_data_spec': {'animal_name': 'V', 'filter_block_type': 'both', 'block_truncation': (10, 70)},
            'behav_format': 'cog_session',
            ### model info
            'agent_type': 'PRLCog',
            'cog_type': 'MB0',
            'seed': 0,
            'model_path': 'test_cogagent_running/test_cog_agent',
            }
      # config = {
      #       'dataset': 'MillerRat',
      #       'behav_data_spec': {'animal_name': 'm55'},
      #       'behav_format': 'cog_session',
      #       ### model info
      #       'agent_type': 'RTSCog',
      #       'cog_type': 'MB0',
      #       'seed': 0,
      #       'model_path': 'test_cogagent_running/test_cog_agent',
      #       }
      # config = {
      #       'dataset': 'AkamRat',
      #       'behav_data_spec': {'animal_name': 267},
      #       'behav_format': 'cog_session',
      #       ### model info
      #       'agent_type': 'NTSCog',
      #       'cog_type': 'MF_MB_bs_rb_ck',
      #       'seed': 0,
      #       'model_path': 'test_cogagent_running/test_cog_agent',
      #       }
      behav_dt = Dataset(config['dataset'], behav_data_spec=config['behav_data_spec'])
      behav_dt = behav_dt.behav_to(config)  # transform format following specifications
      print('Data block num', behav_dt.batch_size)
      behav_data = behav_dt.get_behav_data(np.arange(behav_dt.batch_size), {'behav_format': 'cog_session'})
      ag = Agent(config['agent_type'], config=config)
      ag.save(verbose=True)
      ag.load(config['model_path'])

      output = ag(behav_data['input'])
      print(output.keys(), len(output['output']),len(output['internal']))
