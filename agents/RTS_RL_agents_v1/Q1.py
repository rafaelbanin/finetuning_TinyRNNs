"""Model-free Q(1) model, see Akam et al. (2015) for details."""
import numpy as np
from numba import jit
from random import random, randint
from .core import TwoStepModelCoreCSO, _compute_loglik, _step_other_variables

@jit(nopython=True)
def _Q1_step_core_variables(alpha, c, s, o, Q_td):
    nc = 1 - c  # Not chosen first step action.
    ns = 1 - s  # Not reached second step state.

    Q_td_new = Q_td.copy()
    Q_td_new[c] = (1. - alpha) * Q_td[c] +  alpha * o
    # update action values.

    return Q_td_new


@jit(nopython=True)
def _Q1_session_likelihood_core(alpha, iTemp, choices, second_steps, outcomes, Q_td, scores, choice_probs, n_trials):
    trial_log_likelihood = np.zeros(n_trials)
    for trial in range(n_trials):
        c, s, o = choices[trial], second_steps[trial], outcomes[trial]
        trial_log_likelihood[trial] = _compute_loglik(choice_probs[trial], c)
        Q_td[trial + 1] = _Q1_step_core_variables(alpha, c, s, o, Q_td[trial])
        scores[trial + 1], choice_probs[trial + 1] = _step_other_variables(iTemp, Q_td[trial + 1])
    return trial_log_likelihood, Q_td, scores, choice_probs

class Q1(TwoStepModelCoreCSO):
    def __init__(self):
        super().__init__()
        self.name = 'Q1'
        self.param_names = ['alpha', 'iTemp']
        self.params = [0.5, 5.]
        self.param_ranges = ['unit', 'pos']
        self.n_params = 2
        self.state_vars = ['Q_td']

    def _init_core_variables(self, wm, params):
        if wm is None:
            self.wm = {
                'Q_td': np.zeros(2),
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
        alpha, iTemp = params
        self.wm['Q_td'] = _Q1_step_core_variables(alpha, c, s, o, self.wm['Q_td'])

    def _step_other_variables(self, params):
        alpha, iTemp = params
        self.wm['scores'], self.wm['choice_probs'] = _step_other_variables(iTemp, self.wm['Q_td'])

    def _session_likelihood_core(self, session, params, DVs):
        alpha, iTemp = params
        DVs['trial_log_likelihood'], DVs['Q_td'], DVs['scores'], DVs['choice_probs'] = _Q1_session_likelihood_core(
            alpha, iTemp, session['choices'], session['second_steps'], session['outcomes'],
            DVs['Q_td'], DVs['scores'], DVs['choice_probs'], session['n_trials'])
        return DVs