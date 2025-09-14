"""Model based model. The MB model with nonchosen value decay. """
import numpy as np
from numba import jit
from random import random, randint
from .core import TwoStepModelCoreCSO, _compute_loglik, _step_other_variables

@jit(nopython=True)
def _MB_step_core_variables(alpha, beta, alpha_pers, k_pers, p_transit, c, s, o, Q_s, Q_mb, Pers, Q_tot):
    nc = 1 - c  # Not chosen first step action.
    ns = 1 - s  # Not reached second step state.

    Q_s_new = Q_s.copy()
    Pers_new = Pers.copy()
    # update action values.
    Q_s_new[s] = (1. - alpha) * Q_s[s] + alpha * o
    Q_s_new[ns] = beta * Q_s[ns]

    Q_mb_new = p_transit * Q_s_new + (1-p_transit) * Q_s_new[::-1]

    Pers_new[c] = (1. - alpha_pers) * Pers[c] + alpha_pers * k_pers # action perseveration
    Pers_new[nc] = (1. - alpha_pers) * Pers[nc] - alpha_pers * k_pers
    Q_tot_new = Q_mb_new + Pers_new
    return Q_s_new, Q_mb_new, Pers_new, Q_tot_new


@jit(nopython=True)
def _MB_session_likelihood_core(alpha, beta, alpha_pers, k_pers, iTemp, p_transit, choices, second_steps, outcomes,
                                Q_s, Q_mb, Pers, Q_tot, scores, choice_probs, n_trials):
    trial_log_likelihood = np.zeros(n_trials)
    for trial in range(n_trials):
        c, s, o = choices[trial], second_steps[trial], outcomes[trial]
        trial_log_likelihood[trial] = _compute_loglik(choice_probs[trial], c)
        Q_s[trial + 1], Q_mb[trial + 1], Pers[trial + 1], Q_tot[trial + 1] = _MB_step_core_variables(alpha, beta, alpha_pers, k_pers, p_transit, c, s, o,
                                                                  Q_s[trial], Q_mb[trial], Pers[trial], Q_tot[trial])
        scores[trial + 1], choice_probs[trial + 1] = _step_other_variables(iTemp, Q_tot[trial + 1])
    return trial_log_likelihood, Q_s, Q_mb, Pers, Q_tot, scores, choice_probs

class Model_based_decay_pers(TwoStepModelCoreCSO):
    def __init__(self, p_transit=0.8, use_decay=True):
        super().__init__()
        if use_decay:
            self.name = 'Model based decay persevaration'
            self.param_names = ['alpha', 'beta', 'alpha_pers', 'k_pers', 'iTemp']
            self.params = [ 0.5   , 0.5   ,0.5   , 0,  5.    ]
            self.param_ranges = ['unit' , 'unit' ,'unit' , 'unc' ,'pos'  ]
            self.n_params = 5
        else:
            self.name = 'Model based & persevaration'
            self.param_names = ['alpha', 'alpha_pers', 'k_pers', 'iTemp']
            self.params = [ 0.5   ,0.5   , 0,  5.    ]
            self.param_ranges = ['unit' ,'unit' , 'unc' ,'pos'  ]
            self.n_params = 4

        self.p_transit = p_transit
        self.state_vars = ['Q_s', 'Pers']

    def _init_core_variables(self, wm, params):
        if wm is None:
            self.wm = {
                'Q_s': np.zeros(2),
                'Q_mb': np.zeros(2),
                'Pers': np.zeros(2),
                'Q_tot': np.zeros(2),
            }
        else:
            raise NotImplementedError

    def _step_core_variables(self, trial_event, params):
        (c, s, o) = trial_event
        if len(params) == 5:
            alpha, beta, alpha_pers, k_pers, iTemp = params
        else:
            alpha, alpha_pers, k_pers, iTemp = params
            beta = 1 # no decay
        self.wm['Q_s'], self.wm['Q_mb'], self.wm['Pers'], self.wm['Q_tot'] = _MB_step_core_variables(
            alpha, beta, alpha_pers, k_pers, iTemp, self.p_transit, c, s, o,
            self.wm['Q_s'], self.wm['Q_mb'], self.wm['Pers'], self.wm['Q_tot'])

    def _step_other_variables(self, params):
        if len(params) == 5:
            alpha, beta, alpha_pers, k_pers, iTemp = params
        else:
            alpha, alpha_pers, k_pers, iTemp = params
            beta = 1 # no decay
        self.wm['scores'], self.wm['choice_probs'] = _step_other_variables(iTemp, self.wm['Q_tot'])

    def _session_likelihood_core(self, session, params, DVs):
        if len(params) == 5:
            alpha, beta, alpha_pers, k_pers, iTemp = params
        else:
            alpha, alpha_pers, k_pers, iTemp = params
            beta = 1 # no decay
        DVs['trial_log_likelihood'], DVs['Q_s'], DVs['Q_mb'], DVs['Pers'], DVs['Q_tot'], DVs['scores'], DVs['choice_probs'] = _MB_session_likelihood_core(
            alpha, beta, alpha_pers, k_pers, iTemp, self.p_transit, session['choices'], session['second_steps'], session['outcomes'],
            DVs['Q_s'], DVs['Q_mb'], DVs['Pers'], DVs['Q_tot'],
            DVs['scores'], DVs['choice_probs'], session['n_trials'])
        return DVs