"""
Test the interactions of agents and datasets.
"""
import sys
import time
sys.path.append('..')
from utils import goto_root_dir
from datasets import Dataset
from agents.network_models import MaskedBlockLinear
from agents import RNNAgent
from pathlib import Path
from training_experiments import training
from path_settings import *
from pathlib import Path



# this supports call by both console and main.py
# __name__ ==  '__main__' is required by multiprocessing!
if __name__ ==  '__main__' or '.' in __name__:
      goto_root_dir.run()

      # test MaskedBlockLinear
      a = MaskedBlockLinear(6, 6, 3)
      print(a.weight.shape, a.weight)
      print(a.mask.shape, a.mask)

      ## If testing an initialized rnn agent
      config = {
            ### dataset info; other dataset can be loaded by changing the dataset name
            'dataset': 'BartoloMonkey',
            'behav_data_spec': {'animal_name': 'V', 'filter_block_type': 'both', 'block_truncation': (10, 70)}, # 'both' for all blocks
            # 'dataset': 'MillerRat',
            # 'behav_data_spec': {'animal_name': 'm55', 'block_truncation': (0, 712)},
            # 'dataset': 'AkamRat',
            # 'behav_data_spec': {'animal_name': 267, 'block_truncation': (0, 566)},
            'behav_format': 'tensor',
            ### model info
            'rnn_type': 'MLR', #'GRU',
            'input_dim': 3,
            'hidden_dim': 2,
            'output_dim': 2,
            'expand_size': 10,
            'device': 'cuda', #'cpu',
            'trainable_h0': False,
            'readout_FC': True,
            'one_hot': False,
            'seed': 0,
            'model_path': 'test_agent_running/test_rnn_agent',
            }
      behav_dt = Dataset(config['dataset'], behav_data_spec=config['behav_data_spec'])
      behav_dt = behav_dt.behav_to(config)  # transform format following specifications
      print('Data block num', behav_dt.batch_size)
      behav_data = behav_dt.get_behav_data([0], {'behav_format': 'tensor'})

      rnn = RNNAgent(config)
      # rnn.save(verbose=True)
      # rnn.load(config['model_path'])
      start = time.time()
      output = rnn(behav_data['input'])
      # print(output.keys(), output['output'].shape,output['internal'].shape)
      print('Time elapsed', time.time() - start)