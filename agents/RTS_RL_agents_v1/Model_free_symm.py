"""Model free model. The standard 1-dim MF model. """
import numpy as np
from numba import jit
from random import random, randint
from .core import TwoStepModelCoreCSO, _compute_loglik, _step_other_variables

@jit(nopython=True)
def _MF_step_core_variables(alpha, c, s, o, Q):
    nc = 1 - c  # Not chosen first step action.

    Q_new = Q.copy()
    # update action values.
    Q_new[c] = (1. - alpha) * Q[c] + alpha * o
    Q_new[nc] = (1. - alpha) * Q[nc] - alpha * o

    return Q_new


@jit(nopython=True)
def _MF_session_likelihood_core(alpha, iTemp, choices, second_steps, outcomes, Q, scores, choice_probs, n_trials):
    trial_log_likelihood = np.zeros(n_trials)
    for trial in range(n_trials):
        c, s, o = choices[trial], second_steps[trial], outcomes[trial]
        trial_log_likelihood[trial] = _compute_loglik(choice_probs[trial], c)
        Q[trial + 1] = _MF_step_core_variables(alpha, c, s, o, Q[trial])
        scores[trial + 1], choice_probs[trial + 1] = _step_other_variables(iTemp, Q[trial + 1])
    return trial_log_likelihood, Q, scores, choice_probs

class Model_free_symm(TwoStepModelCoreCSO):
    def __init__(self, equal_reward=False):
        super().__init__()
        self.name = 'Model free symm'
        self.param_names = ['alpha', 'iTemp']
        self.params = [0.5, 5.]
        self.param_ranges = ['unit', 'pos']
        self.n_params = 2
        self.equal_reward = equal_reward
        self.state_vars = ['Q']

    def _init_core_variables(self, wm, params):
        if wm is None:
            self.wm = {
                'Q': np.zeros(2),
            }
        else:
            if 'h0' in wm:
                self.wm = {
                    'Q': wm['h0'],
                }
            else:
                self.wm = wm

    def _step_core_variables(self, trial_event, params):
        (c, s, o) = trial_event
        alpha, iTemp = params
        if self.equal_reward and o == 0:
            o = -1
        self.wm['Q'] = _MF_step_core_variables(alpha, c, s, o, self.wm['Q'])

    def _step_other_variables(self, params):
        alpha, iTemp = params
        self.wm['scores'], self.wm['choice_probs'] = _step_other_variables(iTemp, self.wm['Q'])

    def _session_likelihood_core(self, session, params, DVs):
        alpha, iTemp = params
        outcomes = session['outcomes']
        if self.equal_reward:
             outcomes = outcomes * 2 - 1 # 0 -> -1, 1 -> 1
        DVs['trial_log_likelihood'], DVs['Q'], DVs['scores'], DVs['choice_probs'] = _MF_session_likelihood_core(
            alpha, iTemp, session['choices'], session['second_steps'], outcomes,
            DVs['Q'], DVs['scores'], DVs['choice_probs'], session['n_trials'])
        return DVs