"""The base class for all cognitive agents (RTS, NTS)."""
from .BaseAgent import BaseAgent
import os
import json
import joblib
from path_settings import *
import numpy as np
import random

class CogAgent(BaseAgent):
    """All types of Cog agents.

    Currently all Cog agents come from Akam's two-step task, novel two-step task, and Daw's two-step task.

    Attributes:
        model: the Cog agent, implemented in Akam's way.

    """

    def __init__(self, config=None):
        super().__init__()
        if not hasattr(self, 'config'):
            self.config = config
            self.model = None
            self.cog_type = None
            self.state_vars = []

    @property
    def params(self):
        return self.model.params

    def simulate(self, *args, **kwargs):
        # remove the following arguments, keeping consistency with RNNAgent
        for k in ['update_session', 'pack_trial_event', 'use_one_hot']:
            if k in kwargs:
                kwargs.pop(k)

        # if not hasattr(self.model, params):
        return self.model.simulate(*args, **kwargs)

    def load(self, model_path, strict=True):
        """Load model parameters from disk.
        Args:
            strict: not used. Compatibility with RNNAgent.
        """
        self.set_params(joblib.load(MODEL_SAVE_PATH / model_path / 'model.ckpt'))

    def save(self, params=None, verbose=False):
        """Save config, model, and results."""
        model_path = self.config['model_path']
        os.makedirs(MODEL_SAVE_PATH / model_path, exist_ok=True)
        # save config
        self.save_config()

        # save model parameters
        if params is None:
            joblib.dump(self.params, MODEL_SAVE_PATH / model_path / 'model.ckpt')
        else:
            joblib.dump(params, MODEL_SAVE_PATH / model_path / 'model.ckpt')


        if verbose: print('Saved model at', MODEL_SAVE_PATH / model_path)
        pass

    def forward(self, input, h0=None, standard_output=True,pointwise_loss=False, demask=True):
        """Process a batch of input.

        Args:
            input: List of nn_session instances, each of which contains a block of trials.
                nn_session: Access attributes by session['attr'] (standard format) or session.CSTO['attr'] (old format).
            h0: initial hidden state
            standard_output: Not used. Compatibility with RNNAgent.
            pointwise_loss: Not used. Compatibility with RNNAgent.
            demask: Not used. Compatibility with RNNAgent.
        Returns:
            a dict of internal states and outputs
        """
        if isinstance(input, dict):
            input = input['input']
        nn_sessions = input
        params = self.params
        internal = {}
        scores = []
        masks = []
        behav_loss_sessions = []
        total_corrected_trials = []
        total_behav_loss = 0
        total_trial_num = 0
        total_correct_trials = 0
        for s in nn_sessions:
            mask = None
            if isinstance(s, dict) and 'mask' in s:
                mask = s['mask']
                total_trial_num += s['mask'].sum()
            else:
                mask = np.ones(s.n_trials)
                total_trial_num += s.n_trials if hasattr(s, 'n_trials') else s['n_trials']
            if h0 is not None:
                self.model.init_wm(wm={'h0': h0}, params=params)
            internal_session = self.model.session_likelihood(s, params, get_DVs=True)

            if 'trial_is_correct' in internal_session:
                corrected_trials = internal_session['trial_is_correct']
            else:
                choices = s.CTSO['choices'] if hasattr(s, 'CTSO') else s['choices']
                assert len(internal_session['scores'].shape) == 2
                shape0, shape1 = internal_session['scores'].shape
                if shape0 >= len(choices): # same, or shape0 == len(outcomes) +1
                    action_axis = 1
                elif shape1 >= len(choices): # same, or shape1 == len(outcomes) +1
                    action_axis = 0
                else:
                    raise ValueError(f'The shape of scores {internal_session["scores"].shape} does not match the length of choices {len(choices)}')
                corrected_trials = (np.argmax(internal_session['scores'], axis=action_axis)[:len(choices)] == choices) * 1
            total_corrected_trials.append(corrected_trials)
            total_behav_loss += -internal_session['session_log_likelihood']
            scores.append(internal_session['scores'])
            masks.append(mask)
            behav_loss_sessions.append(internal_session['session_log_likelihood'])
            for k, v in internal_session.items():
                if k not in ['scores', 'session_log_likelihood']:
                    internal.setdefault(k, []).append(v)
            # collect state variables if specified by the model
            state_var_concat = []
            if hasattr(self.model, 'state_vars'):
                for k in self.state_vars:
                    v = internal_session[k]
                    match len(v.shape):
                        case 1:
                            v = v.reshape(v.shape[0], 1)
                        case 2:
                            pass
                        case _:
                            v = v.reshape(v.shape[0], -1)

                    state_var_concat.append(v)
            if len(state_var_concat) > 0:
                state_var = np.concatenate(state_var_concat, axis=1)
            else:
                state_var = np.array([])
            internal.setdefault('state_var', []).append(state_var)
        return {'output': scores, 'internal': internal,
                'behav_loss': total_behav_loss / total_trial_num, 'total_trial_num': total_trial_num, 'total_corrected_trials': total_corrected_trials, 'mask': masks}


    def _set_init_params(self):
        """Set initial parameters for the model.

        This is a helper function. self.model should at least has the param_range attribute, which is a list of range strings.
        Range strings:
            'unit':  # Range: 0 - 1.
            'half':  # Range: 0 - 0.5
            'pos':  # Range: 0 - inf
            'unc': # Range: - inf - inf.
        """

        if hasattr(self.model, 'params'):
            # for some agents, the initial parameters are already set
            pass
        else:
            # for some agents, the initial parameters are not set; inferred them from params_ranges
            self.model.params = []
            for param_range in self.model.param_ranges:
                self.model.params.append({'unit': 0.5, 'pos': 5, 'unc': 0.2, 'half': 0.2}[param_range])
        self.num_params = len(self.params)

    def set_params(self, params):
        """Set model parameters."""
        for i, param in enumerate(params):
            self.model.params[i] = param # make the change in-place


