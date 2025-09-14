""" A wrapper of CogAgent, adding additional functions for training."""
from .CogAgent import CogAgent
import time
import numpy as np
import math
import scipy.optimize as op
import joblib
from path_settings import *

class CogAgentTrainer(CogAgent):
    """
    A wrapper of CogAgent, adding additional functions for training.

    Attributes:
        config: a dict of configuration
        num_params: total number of trainable parameters in the model
        data: data can be directly bound before training (useful when multiprocessing)
    """

    def __init__(self, agent):
        """Initialize the trainer.

        Should not call __init__ of the parent class, since it will initialize a new model.
        Notice here we cannot directly write self.save = agent.save,
        since the agent.save method just becomes a bound method, and the self in the agent.save method is the agent.
        """
        self.agent = agent
        attr_from_agent = ['config', 'model', 'num_params', 'cog_type', 'state_vars', # attributes from the agent
                           'forward', 'load', 'save_config', '_set_init_params' # methods from the agent; save is re-written
                           ]
        for attr in attr_from_agent:
            if hasattr(agent, attr):
                setattr(self, attr, getattr(agent, attr))
        self.save_config()
        self.num_params = len(self.params)


    # multiprocessing has complex interactions with __getattr__, so when uncommenting the following line be careful
    # def __getattr__(self, item):
    #     """If the attribute is not found in the trainer, search in the agent."""
    #     if item == 'agent':
    #         raise ValueError('Encountered infinite recursion. Check the call stack.')
    #     return getattr(self.agent, item)
    # def __getstate__(self):
    #     """Multiprocessing requires the object to be pickable.
    #     This function is called when pickling the object. Create this function to avoid call __getattr__ when pickling."""
    #     return self.__dict__
    # def __setstate__(self, state):
    #     """Multiprocessing requires the object to be pickable.
    #     This function is called when unpickling the object. Create this function to avoid call __getattr__ when unpickling."""
    #     self.__dict__.update(state)

    def bind_data(self, data):
        """Data is bound before call train method. Useful when multiprocessing."""
        self.data = data

    def train(self, data=None, verbose_level=1):
        """Train the model on some training data.

        Args:
            data (dict):
                data['train']: training data, containing input, a list of nn_sessions
                data['val']: validation data, containing input, a list of nn_sessions
                data['test']: test data, containing input, a list of nn_sessions
                data['test'] can be the same as data['val'].
            verbose (bool, optional): whether to print the training progress. Defaults to False.

        Returns:
            self: the trained agent
        """
        time_start = time.time()
        if data is None:
            data = self.data
            assert data is not None
        sessions_train = data['train']['input']
        sessions_val = data['val']['input']
        sessions_test = data['test']['input']

        time_start = time.time()

        assert hasattr(self.model, 'session_likelihood')

        if self.config['cog_type'] == 'BAS':
            session_fit = _fit_sessions_BAS(sessions_train, self.model)
        else:
            session_fit = _fit_sessions(sessions_train, self.model, repeats=1) # repeat is 1 because we will run random seeds in the inner loop in the outer loop
        session_params = session_fit['params']

        self.set_params(session_params)
        self.agent.set_params(session_params) # double setting

        best_model_pass = {}
        best_model_pass['train'] = train_pass = self.forward(sessions_train)
        best_model_pass['val'] = val_pass = self.forward(sessions_val) # val dataset is not used in training here, but for selecting the best seed later
        best_model_pass['test'] = test_pass = self.forward(sessions_test)
        assert np.isclose(train_pass['behav_loss'], -session_fit['likelihood']), (
            self.agent.cog_type, train_pass['behav_loss'], -session_fit['likelihood'], train_pass['total_trial_num'], session_fit['total_trial_num'])

        self.best_model_pass = best_model_pass
        self.save(verbose=False)
        if verbose_level>0: print('Model',Path(self.config['model_path']).name, 'Training done. time cost:',time.time() - time_start,
                                  'train loss:', best_model_pass['train']['behav_loss'],
                                  'val loss:', best_model_pass['val']['behav_loss'],
                                  'test loss:', best_model_pass['test']['behav_loss'])
        return self

    def training_diagnose(self):
        pass

    def save(self, params=None, verbose=False):
        self.agent.save(params=params, verbose=verbose)
        if hasattr(self, 'best_model_pass'):
            if self.config['save_model_pass'] == 'full': # save the full model pass
                save_model_pass = self.best_model_pass
            elif self.config['save_model_pass'] == 'minimal': # only save the likelihoods to save space
                save_model_pass = {
                    'train': {'behav_loss': self.best_model_pass['train']['behav_loss']},
                    'val': {'behav_loss': self.best_model_pass['val']['behav_loss']},
                    'test': {'behav_loss': self.best_model_pass['test']['behav_loss']},
                }
            elif self.config['save_model_pass'] == 'none':
                return
            else:
                raise ValueError('Unknown save_model_pass:', self.config['save_model_pass'])
            joblib.dump(save_model_pass, MODEL_SAVE_PATH / self.config['model_path'] / 'best_pass.pkl')


def _trans_UC(values_U, param_ranges):
    """Transform parameters from unconstrained to constrained space.

    Because the optimisation algorithm is unconstrained, we need to transform the final parameters to constrained space."""
    if param_ranges[0] == 'all_unc':
        return values_U
    values_T = []
    for value, rng in zip(values_U, param_ranges):
        if rng   == 'unit':  # Range: 0 - 1.
            if value < -16.:
                value = -16.
            values_T.append(1./(1. + math.exp(-value)))  # Don't allow values smaller than 1e-
        elif rng   == 'half':  # Range: 0 - 0.5
            if value < -16.:
                value = -16.
            values_T.append(0.5/(1. + math.exp(-value)))  # Don't allow values smaller than 1e-7
        elif rng == 'pos':  # Range: 0 - inf
            if value > 16.:
                value = 16.
            values_T.append(math.exp(value))  # Don't allow values bigger than ~ 1e7.
        elif rng == 'unc': # Range: - inf - inf.
            values_T.append(value)
    return np.array(values_T)


def _fit_sessions(sessions, agent, repeats = 5, brute_init = True, verbose = False):
    """Find maximum likelihood parameter estimates for a list of sessions. Implementation modified from Akam 2015."""

    # RL models do not calculate gradient and require parameter transformation from
    # unconstrained to true space.
    method = 'Nelder-Mead'
    calculates_gradient = False

    # we will find a set of parameters that fits all sessions in the list.
    # session_likelihood will return the sum of likelihood across all trials in one session
    # first sum across sessions, then divide by number of all trials in all sessions to get average
    total_trial_num = np.sum([
        np.sum(s['mask']) if isinstance(s, dict) and 'mask' in s else
        (s.n_trials if hasattr(s, 'n_trials') else s['n_trials']) # Warning: this line did not work for multiple choice states
        for s in sessions])
    # fit_func will take transformed parameters suitable for op.minimize
    fit_func = lambda params: np.sum(
        [-agent.session_likelihood(s, _trans_UC(params, agent.param_ranges)) for s in sessions])/total_trial_num

    fits = []
    for i in range(repeats): # Perform fitting. 

        if agent.n_params <= 3 and i == 0 and brute_init:
           # Initialise minimisation with brute force search.
           ranges = tuple([(-5,5) for i in range(agent.n_params)])
           init_params = op.brute(fit_func, ranges, Ns =  20, 
                                  full_output = True, finish = None)[0]
        else:
            init_params = np.random.normal(0, 3., agent.n_params)

        fits.append(op.minimize(fit_func, init_params, jac = calculates_gradient,
                                method = method, options = {'disp': verbose}))           

    fit = fits[np.argmin([f['fun'] for f in fits])]  # Select best fit out of repeats.

    session_fit = {
        'likelihood' : - fit['fun'],
        'param_names': agent.param_names,
        'total_trial_num': total_trial_num,
        'params': _trans_UC(fit['x'], agent.param_ranges) # Transform parameters back to constrained space
   }

    return session_fit



def _fit_sessions_BAS(sessions, agent):
    """Find maximum likelihood parameter estimates for a list of sessions. Optimized for BAS."""

    total_trial_num = np.sum([s.n_trials for s in sessions])
    fit_func = lambda params: np.sum(
        [-agent.session_likelihood(s, params) for s in sessions]) / total_trial_num
    total_right_action = np.concatenate([s.CTSO['choices'] for s in sessions], 0)
    p = np.sum(total_right_action)/total_trial_num
    assert np.isclose(p, np.mean(total_right_action))
    likelihood = -fit_func([p])
    session_fit = {'likelihood' : likelihood,
                   'param_names': agent.param_names,
                   'total_trial_num': total_trial_num,
                   'params': [p],
                   }

    return session_fit