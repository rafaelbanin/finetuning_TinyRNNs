"""Model mixed model. The standard 1-dim MB + 1-dim MF model. """
import numpy as np
from numba import jit
from random import random, randint
from .core import TwoStepModelCoreCSO, _compute_loglik, _step_other_variables

@jit(nopython=True)
def _MX_step_core_variables(alpha_mf, alpha_mb, w, p_transit, c, s, o, Q, Q_s, Q_mb, Q_mx):
    nc = 1 - c  # Not chosen first step action.
    ns = 1 - s  # Not reached second step state.

    Q_new = Q.copy()
    # update action values.
    Q_new[c] = (1. - alpha_mf) * Q[c] + alpha_mf * o
    Q_new[nc] = (1. - alpha_mf) * Q[nc] - alpha_mf * o

    Q_s_new = Q_s.copy()
    # update action values.
    Q_s_new[s] = (1. - alpha_mb) * Q_s[s] + alpha_mb * o
    Q_s_new[ns] = (1. - alpha_mb) * Q_s[ns] - alpha_mb * o
    Q_mb_new = p_transit * Q_s_new + (1-p_transit) * Q_s_new[::-1]

    # weighted sum of MF and MB.
    Q_mx_new = w * Q_mb_new + (1 - w) * Q_new

    return Q_new, Q_s_new, Q_mb_new, Q_mx_new


@jit(nopython=True)
def _MX_session_likelihood_core(alpha_mf, alpha_mb, w, iTemp, p_transit, choices, second_steps, outcomes, Q, Q_s, Q_mb, Q_mx, scores, choice_probs, n_trials):
    trial_log_likelihood = np.zeros(n_trials)
    for trial in range(n_trials):
        c, s, o = choices[trial], second_steps[trial], outcomes[trial]
        trial_log_likelihood[trial] = _compute_loglik(choice_probs[trial], c)
        Q[trial + 1], Q_s[trial + 1], Q_mb[trial + 1], Q_mx[trial + 1] = _MX_step_core_variables(
            alpha_mf, alpha_mb, w, p_transit, c, s, o, Q[trial], Q_s[trial], Q_mb[trial], Q_mx[trial])
        scores[trial + 1], choice_probs[trial + 1] = _step_other_variables(iTemp, Q_mx[trial + 1])
    return trial_log_likelihood, Q, Q_s, Q_mb, Q_mx, scores, choice_probs

class Model_mixed_symm(TwoStepModelCoreCSO):
    def __init__(self, p_transit=0.8):
        super().__init__()
        self.name = 'Model mixed symm'
        self.param_names = ['alpha_mf', 'alpha_mb', 'w', 'iTemp']
        self.params = [0.5, 0.5, 0.5, 5.]
        self.param_ranges = ['unit', 'unit', 'unit', 'pos']
        self.n_params = 4
        self.p_transit = p_transit
        self.state_vars = ['Q', 'Q_s']

    def _init_core_variables(self, wm, params):
        if wm is None:
            self.wm = {
                'Q': np.zeros(2),
                'Q_s': np.zeros(2),
                'Q_mb': np.zeros(2),
                'Q_mx': np.zeros(2),
            }
        else:
            if 'h0' in wm:
                self.wm = {
                    'Q': wm['h0'],
                    'Q_s': wm['h0'],
                    'Q_mb': self.p_transit * wm['h0'] + (1-self.p_transit) * wm['h0'][::-1],
                    # Q_mx not implemented
                }
            else:
                self.wm = wm

    def _step_core_variables(self, trial_event, params):
        (c, s, o) = trial_event
        alpha_mf, alpha_mb, w, iTemp = params
        self.wm['Q'], self.wm['Q_s'], self.wm['Q_mb'], self.wm['Q_mx'] = _MX_step_core_variables(
            alpha_mf, alpha_mb, w, self.p_transit, c, s, o, self.wm['Q'], self.wm['Q_s'], self.wm['Q_mb'], self.wm['Q_mx'])

    def _step_other_variables(self, params):
        alpha_mf, alpha_mb, w, iTemp = params
        self.wm['scores'], self.wm['choice_probs'] = _step_other_variables(iTemp, self.wm['Q_mx'])

    def _session_likelihood_core(self, session, params, DVs):
        alpha_mf, alpha_mb, w, iTemp = params
        outcomes = session['outcomes']
        DVs['trial_log_likelihood'], DVs['Q'], DVs['Q_s'], DVs['Q_mb'], DVs['Q_mx'], DVs['scores'], DVs['choice_probs'] = _MX_session_likelihood_core(
            alpha_mf, alpha_mb, w, iTemp, self.p_transit, session['choices'], session['second_steps'], outcomes,
            DVs['Q'], DVs['Q_s'], DVs['Q_mb'], DVs['Q_mx'], DVs['scores'], DVs['choice_probs'], session['n_trials'])
        return DVs