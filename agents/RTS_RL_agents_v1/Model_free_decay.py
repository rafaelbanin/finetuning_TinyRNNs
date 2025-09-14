"""Model-free decay models, suitable for multi-arm bandit tasks."""
import numpy as np
from numba import jit
from random import random, randint
from .core import TwoStepModelCoreCSO, _compute_loglik, _step_other_variables, _compute_kl

@jit(nopython=True)
def _MFD_step_core_variables(alpha, beta, c, s, o, Q_td):

    Q_td_new = Q_td.copy()
    Q_td_new *= beta # decay parameters.
    Q_td_new[c] = (1. - alpha) * Q_td[c] +  alpha * o

    return Q_td_new


@jit(nopython=True)
def _MFD_session_likelihood_core(alpha, beta, iTemp, choices, second_steps, outcomes, Q_td, scores, choice_probs, n_trials, target):
    trial_log_likelihood = np.zeros(n_trials)
    for trial in range(n_trials):
        c, s, o = choices[trial], second_steps[trial], outcomes[trial]
        if target is None:
            trial_log_likelihood[trial] = _compute_loglik(choice_probs[trial], c)
        else:
            trial_log_likelihood[trial] = _compute_loglik(choice_probs[trial], target[trial])
        Q_td[trial + 1] = _MFD_step_core_variables(alpha, beta, c, s, o, Q_td[trial])
        scores[trial + 1], choice_probs[trial + 1] = _step_other_variables(iTemp, Q_td[trial + 1])
    return trial_log_likelihood, Q_td, scores, choice_probs

class Model_free_decay(TwoStepModelCoreCSO):
    def __init__(self, n_actions=3, decay=True):
        super().__init__()
        self.name = 'Model-free decay'
        self.decay = decay
        if decay:
            self.param_names = ['alpha', 'beta', 'iTemp']
            self.params = [0.5,0.5, 5.]
            self.param_ranges = ['unit','unit', 'pos']
            self.n_params = 3
        else:
            self.param_names = ['alpha', 'iTemp']
            self.params = [0.5, 5.]
            self.param_ranges = ['unit', 'pos']
            self.n_params = 2
            self.beta = 1
        self.state_vars = ['Q_td']
        self.n_actions = n_actions

    def _init_core_variables(self, wm, params):
        if wm is None:
            self.wm = {
                'Q_td': np.zeros(self.n_actions),
            }
        else:
            if 'h0' in wm:
                self.wm = {
                    'Q_td': wm['h0'],
                }
            else:
                self.wm = wm

    def _step_core_variables(self, trial_event, params):
        (c, s, o) = trial_event
        assert np.all(c == s), "MFD assumes that the first step action is the same as the second step state."
        if self.decay:
            alpha, beta, iTemp = params
        else:
            alpha, iTemp = params
            beta = self.beta
        self.wm['Q_td'] = _MFD_step_core_variables(alpha, beta, c, s, o, self.wm['Q_td'])

    def _step_other_variables(self, params):
        if self.decay:
            alpha, beta, iTemp = params
        else:
            alpha, iTemp = params
            beta = self.beta
        self.wm['scores'], self.wm['choice_probs'] = _step_other_variables(iTemp, self.wm['Q_td'])

    def _session_likelihood_core(self, session, params, DVs):
        if self.decay:
            alpha, beta, iTemp = params
        else:
            alpha, iTemp = params
            beta = self.beta
        assert np.all(session['choices'] == session['second_steps']), "MFD assumes that the first step action is the same as the second step state."
        target = session['target'] if 'target' in session else None
        DVs['trial_log_likelihood'], DVs['Q_td'], DVs['scores'], DVs['choice_probs'] = _MFD_session_likelihood_core(
            alpha, beta, iTemp, session['choices'], session['second_steps'], session['outcomes'],
            DVs['Q_td'], DVs['scores'], DVs['choice_probs'], session['n_trials'], target)
        return DVs