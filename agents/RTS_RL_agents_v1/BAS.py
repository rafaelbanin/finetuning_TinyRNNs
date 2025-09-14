"""Baseline model for the two-step task. """
import numpy as np
from numba import jit
from random import random, randint
from .core import TwoStepModelCoreCSO, _compute_loglik, _step_other_variables, protected_log

@jit(nopython=True)
def _BAS_step_core_variables(p):
    Q = np.zeros(2)
    Q[1] = protected_log(p)
    Q[0] = protected_log(1 - p)
    return Q


@jit(nopython=True)
def _BAS_session_likelihood_core(p, choices, second_steps, outcomes, Q, scores, choice_probs, n_trials):
    trial_log_likelihood = np.zeros(n_trials)
    for trial in range(n_trials):
        c, s, o = choices[trial], second_steps[trial], outcomes[trial]
        trial_log_likelihood[trial] = _compute_loglik(choice_probs[trial], c)
        Q[trial + 1] = _BAS_step_core_variables(p)
        scores[trial + 1], choice_probs[trial + 1] = _step_other_variables(1, Q[trial + 1])
    return trial_log_likelihood, Q, scores, choice_probs

class BAS(TwoStepModelCoreCSO):
    """
    Baseline model for the two-step task. One free parameter, p, the probability of choosing the action.
    """
    def __init__(self):
        super().__init__()
        self.name = 'Baseline'
        self.param_names = ['p']
        self.params = [0.5]
        self.param_ranges = ['unit']
        self.n_params = 1

    def _init_core_variables(self, wm, params):
        if wm is None:
            p, = params
            self.wm = {'Q': _BAS_step_core_variables(p),}
        else:
            if 'h0' in wm:
                raise NotImplementedError
            else:
                self.wm = wm

    def _step_core_variables(self, trial_event, params):
        p, = params
        self.wm['Q'] = _BAS_step_core_variables(p)

    def _step_other_variables(self, params):
        self.wm['scores'], self.wm['choice_probs'] = _step_other_variables(1, self.wm['Q'])

    def _session_likelihood_core(self, session, params, DVs):
        p, = params
        DVs['trial_log_likelihood'], DVs['Q'], DVs['scores'], DVs['choice_probs'] = _BAS_session_likelihood_core(
            p, session['choices'], session['second_steps'], session['outcomes'],
            DVs['Q'],  DVs['scores'], DVs['choice_probs'], session['n_trials'])
        return DVs