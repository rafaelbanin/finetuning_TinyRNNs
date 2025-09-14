"""Model-free decay models, suitable for multi-arm bandit tasks."""
import numpy as np
from numba import jit
from random import random, randint
from .core import TwoStepModelCoreCSO, _compute_loglik, _step_other_variables

@jit(nopython=True)
def _MFD_step_core_variables(alpha, beta, alpha_pers, k_pers, c, s, o, Q_td, Pers, Q_tot):

    Q_td_new = Q_td.copy()
    Pers_new = Pers.copy()
    Q_td_new *= beta # decay parameters.
    Q_td_new[c] = (1. - alpha) * Q_td[c] +  alpha * o
    Pers_new *= (1. - alpha_pers) # decay perseveration
    Pers_new[c] += k_pers # action perseveration
    Q_tot_new = Q_td_new + Pers_new
    return Q_td_new, Pers_new, Q_tot_new


@jit(nopython=True)
def _MFD_session_likelihood_core(alpha, beta, alpha_pers, k_pers, iTemp,
                                 choices, second_steps, outcomes, Q_td, Pers, Q_tot,
                                 scores, choice_probs, n_trials, target):
    trial_log_likelihood = np.zeros(n_trials)
    for trial in range(n_trials):
        c, s, o = choices[trial], second_steps[trial], outcomes[trial]
        if target is None:
            trial_log_likelihood[trial] = _compute_loglik(choice_probs[trial], c)
        else:
            trial_log_likelihood[trial] = _compute_loglik(choice_probs[trial], target[trial])
        Q_td[trial + 1], Pers[trial + 1], Q_tot[trial + 1] = _MFD_step_core_variables(
            alpha, beta, alpha_pers, k_pers, c, s, o, Q_td[trial], Pers[trial], Q_tot[trial])
        scores[trial + 1], choice_probs[trial + 1] = _step_other_variables(iTemp, Q_tot[trial + 1])
    return trial_log_likelihood, Q_td, Pers, Q_tot, scores, choice_probs

class Model_free_decay_pers(TwoStepModelCoreCSO):
    def __init__(self, n_actions=3):
        super().__init__()
        self.name = 'Model-free decay pers'
        self.param_names = ['alpha', 'beta',  'alpha_pers', 'k_pers', 'iTemp']
        self.params = [0.5, 0.5, 0.5, 0, 5.]
        self.param_ranges = ['unit','unit', 'unit', 'unc', 'pos']
        self.n_params = 5

        self.state_vars = ['Q_td', 'Pers']
        self.n_actions = n_actions

    def _init_core_variables(self, wm, params):
        if wm is None:
            self.wm = {
                'Q_td': np.zeros(self.n_actions),
                'Pers': np.zeros(self.n_actions),
                'Q_tot': np.zeros(self.n_actions),
            }
        else:
            if 'h0' in wm:
                self.wm = {
                    'Q_td': wm['h0'],
                    'Pers': np.zeros(self.n_actions),
                    'Q_tot': wm['h0'],
                }
            else:
                self.wm = wm

    def _step_core_variables(self, trial_event, params):
        (c, s, o) = trial_event
        assert np.all(c == s), "MFD assumes that the first step action is the same as the second step state."
        alpha, beta, alpha_pers, k_pers, iTemp = params
        self.wm['Q_td'], self.wm['Pers'], self.wm['Q_tot'] = _MFD_step_core_variables(
            alpha, beta, alpha_pers, k_pers, c, s, o, self.wm['Q_td'], self.wm['Pers'], self.wm['Q_tot'])

    def _step_other_variables(self, params):
        alpha, beta, alpha_pers, k_pers, iTemp = params
        self.wm['scores'], self.wm['choice_probs'] = _step_other_variables(iTemp, self.wm['Q_tot'])

    def _session_likelihood_core(self, session, params, DVs):
        alpha, beta, alpha_pers, k_pers, iTemp = params
        assert np.all(session['choices'] == session['second_steps']), "MFD assumes that the first step action is the same as the second step state."
        target = session['target'] if 'target' in session else None
        DVs['trial_log_likelihood'], DVs['Q_td'], DVs['Pers'], DVs['Q_tot'], DVs['scores'], DVs['choice_probs'] = _MFD_session_likelihood_core(
            alpha, beta, alpha_pers, k_pers, iTemp, session['choices'], session['second_steps'], session['outcomes'],
            DVs['Q_td'], DVs['Pers'], DVs['Q_tot'],
            DVs['scores'], DVs['choice_probs'], session['n_trials'], target)
        return DVs