import scipy.io
import numpy as np
from .BaseTwoStepDataset import BaseTwoStepDataset

from scipy import stats

class BartoloMonkeyDataset(BaseTwoStepDataset):
    """A dataset class for the probalistic reversal learning task with binary actions and rewards.

    Bartolo's two monkeys.
    This is indeed one step task, with states set equal to actions.

    Attributes:
         unique_trial_type: How many possible unique trial observations (actions * rewards = 4 combinations)
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
        self.unique_trial_type = 4 # 2 actions * 2 rewards
        # see data documentation
        self.sessions = {
                'V': ['V20161005','V20160929','V20160930','V20161017'],
                'W': ['W20160112','W20160113','W20160121','W20160122'],
            }
        self.sessions['all'] = self.sessions['V'] + self.sessions['W']
        self.subject_remapping = {'V': 0, 'W': 1}
        super().__init__(data_path, behav_data_spec, neuro_data_spec, verbose=verbose)

    def _detect_trial_type(self):
        """Determine trial type from behavioral data."""
        behav = self.behav
        behav['trial_type'] = []
        for i in range(len(behav['action'])):
            behav['trial_type'].append(behav['action'][i] * 2 + behav['reward'][i])

    def _load_all_trial_type(self, behav_data_spec):
        """Create artificial dataset with all trial types."""
        np2list = lambda L: [np.array([x]) for x in L]
        self.behav = {
            'action': np2list([0, 0, 1, 1]),
            'stage2': np2list([0, 0, 1, 1]),
            'reward': np2list([0, 1, 0, 1]),
        }
        self._detect_trial_type()

    def load_data(self, behav_data_spec, neuro_data_spec=None, verbose=True):
        """Load behavioral and neural data.

        For the meaning of each variable see DATA_DOCUMENTATION.pdf in data folder.
        The loaded data is stored in self.behav, the standard format of behavioral data.

        Args:
            behav_data_spec: A dictionary of behavioral data specification.
            neuro_data_spec: A dictionary of neural data specification.
                select_bins
         """
        if 'all_trial_type' in behav_data_spec and behav_data_spec['all_trial_type']:
            self._load_all_trial_type(behav_data_spec)
            return
        filter_block_type = behav_data_spec['filter_block_type']
        assert filter_block_type in ['both', 'what', 'where']
        if 'animal_name' in behav_data_spec and 'session_name' not in behav_data_spec:
            behav_data_spec['session_name'] = self.sessions[behav_data_spec['animal_name']]
        if isinstance(behav_data_spec['session_name'], str):
            behav_data_spec['session_name'] = [behav_data_spec['session_name']]
        if neuro_data_spec is None:
            neuro_data_spec = {}
        self.behav_data_spec = behav_data_spec
        self.neuro_data_spec = neuro_data_spec

        self.behav = behav = {}
        self.neuro = neuro = {}

        if 'include_task' in behav_data_spec and behav_data_spec['include_task']:
            assert 'sub_index_start' in behav_data_spec # external scalar
            assert 'sub_index_end' in behav_data_spec # external scalar
            sub_index_start = behav_data_spec['sub_index_start'] # left inclusive
            sub_index_end = behav_data_spec['sub_index_end'] # right exclusive
            assert sub_index_end - sub_index_start == len(self.subject_remapping), f'{sub_index_end} - {sub_index_start} != {len(self.subject_remapping)}'
            task_id = behav_data_spec['task_id']
        else:
            sub_index_start = 0
            sub_index_end = len(self.subject_remapping)
            task_id = 0

        for sess_name in behav_data_spec['session_name']:
            if verbose: print('====Load session:', sess_name, 'filter_block_type:', filter_block_type)
            sub_id = self.subject_remapping[sess_name[0]] + sub_index_start
            mat = scipy.io.loadmat(self.data_path / f'SPKcounts_{sess_name}cue_MW_250X250ms.mat')
            spikes_all = mat['X']  # neuron_num, total_trial_num, bin_num
            if 'select_bins' in neuro_data_spec:
                select_bins = neuro_data_spec['select_bins']
                if verbose: print('Select bins:', select_bins)
                spikes_all = spikes_all[..., select_bins]
            if 'combine_bins' in neuro_data_spec and neuro_data_spec['combine_bins']:
                spikes_all = spikes_all.sum(-1)[..., None]
            chosen_image = mat['Y'][:, 0].astype('int64')
            chosen_loc = mat['Y'][:, 1].astype('int64')
            outcome = mat['Y'][:, 2].astype('int64')
            complete_trial = mat['Y'][:, 3]
            best_chosen = mat['Y'][:, 4]
            trial_in_block = mat['Y'][:, 5]
            reversal_trial = mat['Y'][:, 6]
            block_id = mat['Y'][:, 7]
            blockorder = mat['Y'][:, 8]
            block_type = mat['Y'][:, 9]
            CorrectTrialnumberinsession = mat['Y'][:, 10]
            TotalTrialnumberinblock = mat['Y'][:, 11]
            BlockCompleted = mat['Y'][:, 12]
            ReactTargetRewardTimes = mat['ReactTargetRewardTimes'] # total_trial_num, 3
            cell_num, total_trial_num, bin_num = spikes_all.shape
            X = np.swapaxes(spikes_all, 0, 1)  # total_trial_num, cell_num, bin_num
            X = X.reshape((total_trial_num, cell_num * bin_num))
            self.bin_num = bin_num
            trial_num = 80
            if filter_block_type in ['what', 'where']:
                if filter_block_type == 'what':
                    block_type_idx = 1
                    act = chosen_image # use the chosen image as action
                    if verbose: print('WARNING: loading what blocks')
                else:
                    block_type_idx = 2
                    act = chosen_loc # use the chosen location as action
                trial_filter = np.all([block_type == block_type_idx, block_id >= 1, block_id <= 24, BlockCompleted], axis=0)

            elif filter_block_type == 'both':
                act = chosen_image.copy() # use the chosen image as action, then replace some actions with chosen location
                trial_filter2 = np.all([block_type == 2, block_id >= 1, block_id <= 24, BlockCompleted], axis=0)
                act[trial_filter2] = chosen_loc[trial_filter2]  # replace some actions

                trial_filter = np.all([block_id >= 1, block_id <= 24, BlockCompleted], axis=0)

            else:
                raise ValueError

            assert trial_filter.sum() % trial_num == 0
            episode_num = trial_filter.sum() // trial_num
            act = act[trial_filter].reshape((episode_num, trial_num))
            reward = outcome[trial_filter].reshape((episode_num, trial_num))
            reversal_trial = reversal_trial[trial_filter].reshape((episode_num, trial_num))
            X = X[trial_filter].reshape((episode_num, trial_num, cell_num * bin_num))
            reaction_time = ReactTargetRewardTimes[trial_filter, 0].reshape((episode_num, trial_num))
            target_time = ReactTargetRewardTimes[trial_filter, 1].reshape((episode_num, trial_num))
            reward_time = ReactTargetRewardTimes[trial_filter, 2].reshape((episode_num, trial_num))
            block_type = block_type[trial_filter].reshape((episode_num, trial_num)).mean(1).astype('int64')
            if 'block_truncation' in behav_data_spec:
                bt = behav_data_spec['block_truncation']
                if verbose: print('block_truncation', bt)
                eff_trials = np.arange(bt[0], bt[1])
                X = X[:, eff_trials]
                act = act[:, eff_trials]
                reward = reward[:, eff_trials]
                reaction_time = reaction_time[:, eff_trials]
                target_time = target_time[:, eff_trials]
                reward_time = reward_time[:, eff_trials]
                reversal_trial = reversal_trial[:, eff_trials]

            if verbose:
                print('Spikes: episode num, trial num, feature_num =', X.shape)
                print('Observations: episode num, trial num =', act.shape)
                print('mean reward', reward.mean())

            behav.setdefault('action', []).extend(list(act))  # list of 1d array
            behav.setdefault('stage2', []).extend(list(act))
            behav.setdefault('reward', []).extend(list(reward))
            behav.setdefault('reversal_trial', []).extend(list(reversal_trial))
            behav.setdefault('sub_id', []).extend([sub_id] * len(act))
            behav.setdefault('task_id', []).extend([task_id] * len(act))

            behav.setdefault('reaction_time', []).extend(list(reaction_time))
            behav.setdefault('target_time', []).extend(list(target_time))
            behav.setdefault('reward_time', []).extend(list(reward_time))

            neuro.setdefault('block_type', []).extend(list(np.array(['what', 'where'])[block_type - 1]))
            neuro.setdefault('session_name', []).extend([sess_name] * episode_num)
            neuro.setdefault('X', []).extend(list(X))

            if 'preprocess' in neuro_data_spec:
                X_preprocessed, _, _ = self.preprocess_X(X,
                                                                 zcore=neuro_data_spec['preprocess']['zcore'],
                                                                 remove_nan=neuro_data_spec['preprocess']['remove_nan'],
                                                                 pca=neuro_data_spec['preprocess']['pca'])
                neuro.setdefault('X_preprocessed', []).extend(list(X_preprocessed))

        self._detect_trial_type()
        print("===loaded all===", 'Bartolo Monkey')
        print('Total batch size:', self.batch_size)
        print('Total trial num:', self.total_trial_num)

    def get_neuro_data(self, session_name='', block_type='', zcore=True, remove_nan=True, shape=3, pca=-1, **kwargs):
        """Get neural data for a session and block type.
        Note: before calling this function, each block in behav and each block in neuro are one-to-one mapping (i.e. same order).
        The returned episodic data is the index of X blocks from the behav blocks.
        Args:
            session_name: str
            block_type: str
            zcore: bool
            remove_nan: bool
            shape: int, 2 for (total_trial_num, feature_num), 3 for (episode_num, trial_num, feature_num)

        Returns:
            X: np.ndarray, shape (episode_num, trial_num, feature_num)
            episode_idx: np.ndarray, shape episode_num; indices for the selected episodes
            kept_feat_idx: np.ndarray, shape feature_num; indices for the selected features (non-nan)
            feat_scale: np.ndarray, shape feature_num; scale for each feature due to z-scoring
        """
        assert session_name in self.neuro['session_name']
        assert block_type in ['what', 'where', 'both']
        assert shape in [2, 3]
        sess_idx = np.array(self.neuro['session_name']) == session_name
        if block_type == 'both':
            block_type_idx = np.ones_like(sess_idx)
        else:
            block_type_idx = np.array(self.neuro['block_type']) == block_type
        episode_idx = np.where(sess_idx & block_type_idx)[0]
        X = np.array([self.neuro['X'][i] for i in episode_idx])

        X, kept_feat_idx, feat_scale = self.preprocess_X(X, zcore=zcore, remove_nan=remove_nan, pca=pca)
        if shape == 2:
            X = X.reshape((-1, X.shape[-1])) # (total_trial_num, feat_num)

        return X, episode_idx, kept_feat_idx, feat_scale

    def preprocess_X(self, X, zcore=True, remove_nan=True, pca=-1):
        # pca: -1 for no pca processing
        # 0 for empty X
        # n for pca to n dimension
        episode_num, trial_num, feat_num = X.shape
        if zcore: # zscore for each feature
            feat_scale = np.std(X.reshape((episode_num*trial_num,feat_num)), axis=0)
            X = stats.zscore(X.reshape((episode_num*trial_num,feat_num))).reshape((episode_num, trial_num, feat_num))
        else:
            feat_scale = np.ones(feat_num)
        if remove_nan:
            X = X.reshape((episode_num*trial_num,feat_num))
            kept_feat_idx = ~np.isnan(X).any(axis=0)
            X = X[:, kept_feat_idx]  # remove nan columns
            X = X.reshape((episode_num, trial_num, -1))
            feat_scale = feat_scale[kept_feat_idx]
        else:
            kept_feat_idx = np.ones(feat_num, dtype=bool)
        kept_feat_idx = kept_feat_idx.reshape((-1, self.bin_num))

        if pca == -1:
            pass
        elif pca == 0:
            X = np.zeros((episode_num, trial_num, 0))
        elif pca > 0:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=pca)
            X = pca.fit_transform(X.reshape((episode_num*trial_num, -1))).reshape((episode_num, trial_num, -1))
        else:
            raise ValueError('Unknown pca value')
        return X, kept_feat_idx, feat_scale