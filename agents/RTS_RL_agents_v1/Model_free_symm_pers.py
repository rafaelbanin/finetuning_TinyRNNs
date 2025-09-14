"""Model free model with persevaration. The standard 1-dim MF + 1-dim pers. """
import numpy as np
from numba import jit
from random import random, randint
from .core import TwoStepModelCoreCSO, _compute_loglik, _step_other_variables

@jit(nopython=True)
def _MF_step_core_variables(alpha, alpha_pers, k_pers, c, s, o, Q, Pers, Q_tot):
    nc = 1 - c  # Not chosen first step action.

    Q_new = Q.copy()
    Pers_new = Pers.copy()
    # update action values.
    Q_new[c] = (1. - alpha) * Q[c] + alpha * o
    Q_new[nc] = (1. - alpha) * Q[nc] - alpha * o
    Pers_new[c] = (1. - alpha_pers) * Pers[c] + alpha_pers * k_pers # action perseveration
    Pers_new[nc] = (1. - alpha_pers) * Pers[nc] - alpha_pers * k_pers
    Q_tot_new = Q_new + Pers_new
    return Q_new, Pers_new, Q_tot_new


@jit(nopython=True)
def _MF_session_likelihood_core(alpha, alpha_pers, k_pers, iTemp, choices, second_steps, outcomes, Q, Pers, Q_tot, scores, choice_probs, n_trials):
    trial_log_likelihood = np.zeros(n_trials)
    for trial in range(n_trials):
        c, s, o = choices[trial], second_steps[trial], outcomes[trial]
        trial_log_likelihood[trial] = _compute_loglik(choice_probs[trial], c)
        Q[trial + 1], Pers[trial + 1], Q_tot[trial + 1] = _MF_step_core_variables(alpha, alpha_pers, k_pers, c, s, o, Q[trial], Pers[trial], Q_tot[trial])
        scores[trial + 1], choice_probs[trial + 1] = _step_other_variables(iTemp, Q_tot[trial + 1])
    return trial_log_likelihood, Q, Pers, Q_tot, scores, choice_probs

class Model_free_symm_pers(TwoStepModelCoreCSO):
    def __init__(self):
        super().__init__()
        self.name = 'Model free symm pers'
        self.param_names = ['alpha', 'alpha_pers', 'k_pers', 'iTemp']
        self.params = [0.5, 0.5, 0,  5.]
        self.param_ranges = ['unit', 'unit', 'unc', 'pos']
        self.n_params = 4
        self.state_vars = ['Q', 'Pers']

    def _init_core_variables(self, wm, params):
        if wm is None:
            self.wm = {
                'Q': np.zeros(2),
                'Pers': np.zeros(2),
                'Q_tot': np.zeros(2),
            }
        else:
            if 'h0' in wm:
                h0 = wm['h0']

                self.wm = {
                    'Q': np.array([h0[0], -h0[0]]),
                    'Pers': np.array([h0[1], -h0[1]]),
                    'Q_tot': np.array([h0[0] + h0[1], -(h0[0] + h0[1])]),
                }
            else:
                self.wm = wm

    def _step_core_variables(self, trial_event, params):
        (c, s, o) = trial_event
        alpha, alpha_pers, k_pers, iTemp = params
        self.wm['Q'], self.wm['Pers'], self.wm['Q_tot'] = _MF_step_core_variables(
            alpha, alpha_pers, k_pers, c, s, o, self.wm['Q'], self.wm['Pers'], self.wm['Q_tot'])

    def _step_other_variables(self, params):
        alpha, alpha_pers, k_pers, iTemp = params
        self.wm['scores'], self.wm['choice_probs'] = _step_other_variables(iTemp, self.wm['Q_tot'])

    def _session_likelihood_core(self, session, params, DVs):
        alpha, alpha_pers, k_pers, iTemp = params
        outcomes = session['outcomes']
        DVs['trial_log_likelihood'], DVs['Q'], DVs['Pers'], DVs['Q_tot'], DVs['scores'], DVs['choice_probs'] = _MF_session_likelihood_core(
            alpha, alpha_pers, k_pers, iTemp, session['choices'], session['second_steps'], session['outcomes'],
            DVs['Q'], DVs['Pers'], DVs['Q_tot'],
            DVs['scores'], DVs['choice_probs'], session['n_trials'])
        return DVs