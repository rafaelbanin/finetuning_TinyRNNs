"""Model-free Q(0) model, see Akam et al. (2015) for details."""
import numpy as np
from numba import jit
from random import random, randint
from .core import TwoStepModelCoreCSO, _compute_loglik, _step_other_variables

@jit(nopython=True)
def _Q0_step_core_variables(alpha, c, s, o, Q_td_f, Q_td_s):
    nc = 1 - c  # Not chosen first step action.
    ns = 1 - s  # Not reached second step state.

    Q_td_f_new = Q_td_f.copy()
    Q_td_s_new = Q_td_s.copy()
    # update action values.
    Q_td_f_new[c] = (1. - alpha) * Q_td_f[c] + alpha * Q_td_s[s]
    Q_td_s_new[s] = (1. - alpha) * Q_td_s[s] + alpha * o

    return Q_td_f_new, Q_td_s_new


@jit(nopython=True)
def _Q0_session_likelihood_core(alpha, iTemp, choices, second_steps, outcomes, Q_td_f, Q_td_s, scores, choice_probs, n_trials):
    trial_log_likelihood = np.zeros(n_trials)
    for trial in range(n_trials):
        c, s, o = choices[trial], second_steps[trial], outcomes[trial]
        trial_log_likelihood[trial] = _compute_loglik(choice_probs[trial], c)
        Q_td_f[trial + 1], Q_td_s[trial + 1] = _Q0_step_core_variables(alpha, c, s, o, Q_td_f[trial], Q_td_s[trial])
        scores[trial + 1], choice_probs[trial + 1] = _step_other_variables(iTemp, Q_td_f[trial + 1])
    return trial_log_likelihood, Q_td_f, Q_td_s, scores, choice_probs

class Q0(TwoStepModelCoreCSO):
    def __init__(self):
        super().__init__()
        self.name = 'Q0'
        self.param_names = ['alpha', 'iTemp']
        self.params = [0.5, 5.]
        self.param_ranges = ['unit', 'pos']
        self.n_params = 2
        self.state_vars = ['Q_td_f', 'Q_td_s']

    def _init_core_variables(self, wm, params):
        if wm is None:
            self.wm = {
                'Q_td_f': np.zeros(2),
                'Q_td_s': np.zeros(2),
            }
        else:
            if 'h0' in wm:
                raise NotImplementedError
            else:
                self.wm = wm

    def _step_core_variables(self, trial_event, params):
        (c, s, o) = trial_event
        alpha, iTemp = params
        self.wm['Q_td_f'], self.wm['Q_td_s'] = _Q0_step_core_variables(alpha, c, s, o, self.wm['Q_td_f'], self.wm['Q_td_s'])

    def _step_other_variables(self, params):
        alpha, iTemp = params
        self.wm['scores'], self.wm['choice_probs'] = _step_other_variables(iTemp, self.wm['Q_td_f'])

    def _session_likelihood_core(self, session, params, DVs):
        alpha, iTemp = params
        DVs['trial_log_likelihood'], DVs['Q_td_f'], DVs['Q_td_s'], DVs['scores'], DVs['choice_probs'] = _Q0_session_likelihood_core(
            alpha, iTemp, session['choices'], session['second_steps'], session['outcomes'],
            DVs['Q_td_f'], DVs['Q_td_s'], DVs['scores'], DVs['choice_probs'], session['n_trials'])
        return DVs
