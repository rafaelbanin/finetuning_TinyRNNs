import numpy as np
import joblib
import json
import torch
from .dataset_utils import Dataset
from .BaseTwoStepDataset import BaseTwoStepDataset, _combine_data_dict
class MiscDataset(BaseTwoStepDataset):
    """A dataset class to provide APIs for loading and processing data from different agents.
    Currently, only support the reduced two-step task.

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
        assert 'misc' in behav_data_spec
        assert neuro_data_spec is None
        self.behav = {
            'action': [],
            'stage2': [],
            'reward': [],
            'trial_type': [],
            'label': [],
        }
        self.torch_beahv_input = None
        self.torch_beahv_target = None
        self.torch_beahv_mask = None
        self.cog_sessions = None
        self.behav_format = None
        self.load_data(behav_data_spec, neuro_data_spec=None)


    def load_data(self, behav_data_spec, neuro_data_spec=None, verbose=True):
        behav = self.behav
        for each_behav_data_spec in behav_data_spec['misc']:
            print('Loading', each_behav_data_spec)
            behav_dt = Dataset(each_behav_data_spec['dataset'], behav_data_spec=each_behav_data_spec, verbose=verbose)
            for k in ['action', 'stage2', 'reward', 'trial_type']:
                behav[k] += behav_dt.behav[k]
            behav['label'] += [each_behav_data_spec['label']] * len(behav_dt.behav['action'])

        if verbose: print('Misc Total trial num:', self.total_trial_num)

    def _behav_to_tensor(self, format_config):
        """Transform standard behavioral format to tensor format, stored in torch_beahv_* attribute.

        standard format (list of 1d array) -> tensor format (2d array with 0 padding).
        The attributes are:
            torch_beahv_input: tensor format of agent's input
            torch_beahv_input_1hot: tensor format of agent's input (one-hot encoding of trial observations)
            torch_beahv_target: tensor format of agent's target output
            torch_beahv_mask: tensor format of agent's mask (1 for valid trials, 0 for padding trials)

        Not use nan padding:
            rnn model make all-nan output randomly (unexpected behavior, not sure why)
            the one_hot function cannot accept nan
            long type is required for cross entropy loss, but does not support nan value

        Args:
            format_config: A dict specifies how the standard data should be transformed.

        """
        if self.torch_beahv_input is not None:
            return
        max_trial_num = max([len(block) for block in self.behav['reward']])

        act = np.zeros((self.batch_size, max_trial_num))
        rew = np.zeros((self.batch_size, max_trial_num))
        stage2 = np.zeros((self.batch_size, max_trial_num))
        label = np.zeros((self.batch_size, max_trial_num))
        mask = np.zeros((self.batch_size, max_trial_num))
        for b in range(self.batch_size):
            this_trial_num = len(self.behav['reward'][b])
            mask[b, :this_trial_num] = 1
            act[b, :this_trial_num] = self.behav['action'][b]
            rew[b, :this_trial_num] = self.behav['reward'][b]
            stage2[b, :this_trial_num] = self.behav['stage2'][b]
            label[b, :] = self.behav['label'][b]

        device = 'cpu' if 'device' not in format_config else format_config['device']
        output_h0 = True if 'output_h0' not in format_config else format_config['output_h0']

        act = torch.from_numpy(act.T[..., None]).to(device=device)  # act shape: trial_num, batch_size, 1
        stage2 = torch.from_numpy(stage2.T[..., None]).to(device=device)
        rew = torch.from_numpy(rew.T[..., None]).to(device=device)
        target1 = torch.from_numpy(label.T).to(device=device) # label shape: trial_num, batch_size
        if output_h0:
            input = torch.cat([act, stage2, rew], -1)  # trial_num, batch_size, input_size=3
            target = act[:, :, 0]  # class, not one-hot
        else:
            assert NotImplementedError

        self.torch_beahv_input = input.double()
        self.torch_beahv_target = target.long()
        self.torch_beahv_target1 = target1.long()
        self.torch_beahv_mask = torch.from_numpy(mask.T).to(device=device).double()

    def get_behav_data(self, batch_indices, format_config=None, remove_padding_trials=True):
        """Return a subset (a few batches/blocks) of loaded data with specified format.

        Args:
            batch_indices: A list or 1d ndarray specifies which batches are extracted.
            format_config: A dict specifies how the standard data should be transformed.
                Mainly "behav_format" and "one_hot" keys.

        Returns:
            A dict contains:
                input, target_output, mask, if for the tensor format.
                List of sessions (a session = a block), if for the cog_session format.
        """
        assert isinstance(batch_indices, list) or isinstance(batch_indices, np.ndarray) and len(batch_indices.shape) == 1
        assert self.behav_format == 'tensor'
        assert format_config is not None
        assert self.behav_format == format_config['behav_format'] # the data format should first be transformed to the specified one
        if 'trial_type' in self.behav:
            trial_type = np.array(self.behav['trial_type'], dtype=object)
        else:
            print('Warning: no trial_type in the loaded data, set to -1')
            trial_type = np.ones(self.total_trial_num)*-1


        if isinstance(self.torch_beahv_mask, tuple):
            mask = self.torch_beahv_mask[0]
        else:
            mask = self.torch_beahv_mask
        # mask has two purposes:
        # 1. mask out invalid trials (especially for actions)
        # 2. mask out padding trials (for RNN models) <- which is used here
        # however, sometimes the last few trials are invalid, but not padding trials; we don't want to remove them here
        if remove_padding_trials:
            mask = mask[:,batch_indices]
            mask_block_sum = mask.sum(dim=1) # shape (trial_num,)
            last_1_idx = torch.where(mask_block_sum>0)[0][-1].item()
            trial_slice = slice(0, last_1_idx+1) # remove the last few trials with mask=0 for the whole batch
        else:
            trial_slice = slice(0, mask.shape[0])
        assert not ('one_hot' in format_config and format_config['one_hot'])
        assert 'label_prop' in format_config
        return {'input': self.torch_beahv_input[trial_slice,batch_indices],
                'target': [
                    ('action', 1 - format_config['label_prop'], self.torch_beahv_target[trial_slice,batch_indices]),
                    ('label', format_config['label_prop'], self.torch_beahv_target1[trial_slice,batch_indices]),
                ],
                'mask': self.torch_beahv_mask[trial_slice, batch_indices] if not isinstance(self.torch_beahv_mask, tuple)
                    else (self.torch_beahv_mask[0][trial_slice,batch_indices], self.torch_beahv_mask[1][trial_slice,batch_indices]),
                'trial_type': trial_type[batch_indices],
                'label': self.behav['label'],
                }
