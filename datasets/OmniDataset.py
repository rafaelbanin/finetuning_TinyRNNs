import numpy as np
from .BaseTwoStepDataset import BaseTwoStepDataset, _combine_data_dict, _segment_long_block
from . import Dataset
import sys
import joblib

class OmniDataset(BaseTwoStepDataset):
    """A dataset class for three tasks with binary actions, states, and rewards.

    Akam's rats.
    Load preprocessed data generated from his Python code. See analysis_code/two_step/Two_step.py.

    Attributes:
         unique_trial_type: How many possible unique trial observations (actions * states * rewards = 8 combinations)
         behav: Standard format of behavioral data.
         data_path: Where to load the data.
         behav_format: tensor (for RNN) or cog_session (for Cog agents)?
         torch_beahv_input: tensor format of agent's input
         torch_beahv_input_1hot: tensor format of agent's input (one-hot encoding of trial observations)
         torch_beahv_target: tensor format of agent's target output
         cog_sessions: cog_session format of agent's input & target output
         batch_size: How many blocks are there in the current loaded data?
    """
    def __init__(self, data_path=None, behav_data_spec=None, neuro_data_spec=None, verbose=False):
        """Initialize the dataset."""
        data_path = ''
        self.unique_trial_type = -1
        super().__init__(data_path, behav_data_spec, neuro_data_spec, verbose=verbose)

    def load_data(self, behav_data_spec, neuro_data_spec=None, verbose=True):
        """Load data from disk following data specifications."""

        self.dts = dts = []
        behav_data_spec_common = behav_data_spec
        # Akamrat
        sub_index_start = 0
        sub_index_end = 17
        task_id = 0
        behav_data_spec = behav_data_spec_common | {
            'task_id': task_id, 'sub_index_start': sub_index_start, 'sub_index_end': sub_index_end,
            }
        dt = Dataset('AkamRat', behav_data_spec=behav_data_spec, verbose=False)
        dts.append(dt)

        # Akamratprl
        sub_index_start = sub_index_end
        sub_index_end = sub_index_start + 10
        task_id += 1
        behav_data_spec = behav_data_spec_common | {'task': 'reversal_learning',
                           'task_id': task_id, 'sub_index_start': sub_index_start, 'sub_index_end': sub_index_end,
                           }
        dt = Dataset('AkamRat', behav_data_spec=behav_data_spec, verbose=False)
        dts.append(dt)

        # Akamratrts
        sub_index_start = sub_index_end
        sub_index_end = sub_index_start + 10
        task_id += 1
        behav_data_spec = behav_data_spec_common| {'task': 'no_transition_reversal',
                           'task_id': task_id, 'sub_index_start': sub_index_start, 'sub_index_end': sub_index_end,
                           }
        dt = Dataset('AkamRat', behav_data_spec=behav_data_spec, verbose=False)
        dts.append(dt)

        #Monkeys
        sub_index_start = sub_index_end
        sub_index_end = sub_index_start + 2
        task_id += 1
        behav_data_spec = behav_data_spec_common | {'filter_block_type': 'both', 'block_truncation': (10, 70),
                           'task_id': task_id, 'sub_index_start': sub_index_start, 'sub_index_end': sub_index_end,
                           }
        dt = Dataset('BartoloMonkey', behav_data_spec=behav_data_spec, verbose=False)
        dts.append(dt)
        #
        # Millerrat
        sub_index_start = sub_index_end
        sub_index_end = sub_index_start + 5
        task_id += 1
        behav_data_spec = behav_data_spec_common | {
                    'task_id': task_id, 'sub_index_start': sub_index_start, 'sub_index_end': sub_index_end,
                           }
        dt = Dataset('MillerRat', behav_data_spec=behav_data_spec, verbose=False)
        dts.append(dt)

        self.behav = behav = {}
        for key in ['action', 'reward', 'stage2', 'task_id', 'sub_id']:
            behav[key] = []
            for dt in dts:
                behav[key].extend(dt.behav[key])


        print("===loaded all===", 'Everything')
        print('Total batch size:', self.batch_size)
        print('Total trial num:', self.total_trial_num)