import numpy as np
import joblib
import json
from .BaseTwoStepDataset import BaseTwoStepDataset, _combine_data_dict

class MetaTwoStepDataset(BaseTwoStepDataset):
    """A dataset class currently only for meta-RL agent on two-step task.

    Attributes:
         unique_trial_type: How many possible unique trial observations (actions * states * rewards)?
         behav: Standard format of behavioral data.
         data_path: Where to load the data.
         behav_format: tensor (for RNN) or cog_session (for Cog agents)?
         torch_beahv_input: tensor format of agent's input
         torch_beahv_input_1hot: tensor format of agent's input (one-hot encoding of trial observations)
         torch_beahv_target: tensor format of agent's target output
         cog_sessions: cog_session format of agent's input & target output
         batch_size: How many blocks are there in the current loaded data?
    """
    def __init__(self, data_path=None, behav_data_spec=None, neuro_data_spec=None, verbose=True):
        super().__init__(data_path, behav_data_spec, neuro_data_spec, verbose=verbose)

    def load_data(self, behav_data_spec, neuro_data_spec=None, verbose=True):
        file_path = self.data_path
        if 'seed' not in behav_data_spec:
            seed = 0
        else:
            seed = behav_data_spec['seed']
        if 'savedpoint' not in behav_data_spec or behav_data_spec['savedpoint'] == -1:
            savedpoint = 1999
        else:
            savedpoint = behav_data_spec['savedpoint']
        file_path = file_path / f'determin_True' / f'A2C_multistep_jwang_delay1True_seed{seed}' / f'savedpoint_{savedpoint}_behav'
        behav = self.behav = joblib.load(file_path)
        neuro = self.neuro = {}
        behav['gt'] -= 1 # 1->0, 2->1, shape: episode_num, trial_num
        behav['action'] -= 1 # shape: episode_num, trial_num
        behav['stage2'] = (behav['stage2'] == 'B2').astype(int) # 'B1'->0, 'B2'->1 shape: episode_num, trial_num
        neuro['activity'] = behav.pop('activity') # shape: episode_num, trial_num, 3, neuron_num
        neuro['policy_probas'] = behav.pop('policy_probas') # shape: episode_num, trial_num, 3, 3=(F, L, R)
        neuro['policy_logits'] = behav.pop('policy_logits') # shape: episode_num, trial_num, 3, 3=(F, L, R)

        LS = behav['agents_mid_vars']['LSSprob']
        MB = behav['agents_mid_vars']['MBprob']
        neuro['LS_policy_probas'] = LS['choice_probs'] # shape: episode_num, trial_num, 2=(L, R)
        neuro['LS_policy_logits'] = np.log(LS['choice_probs']) # shape: episode_num, trial_num, 2=(L, R)
        neuro['MB_policy_probas'] = MB['choice_probs'] # shape: episode_num, trial_num, 2=(L, R)
        neuro['MB_policy_logits'] = np.log(MB['choice_probs']) # shape: episode_num, trial_num, 2=(L, R)
        behav['trial_type'] = behav['action'] * 4 + behav['stage2'] * 2 + behav['reward']

        # behav.pop('transition')
        # behav.pop('agents_mid_vars')
        self.unique_trial_type = 8
        if verbose: print('Total trial num:', self.total_trial_num)
