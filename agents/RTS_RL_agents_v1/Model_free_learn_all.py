"""Model-free learn-all models, suitable for multi-arm bandit tasks."""
import numpy as np
from numba import jit
from random import random, randint
from .core import TwoStepModelCoreCSO, _compute_loglik, _step_other_variables

@jit(nopython=True)
def _MFL_step_core_variables(alpha_c, offset_c, beta_c, alpha_u, offset_u, beta_u, c, s, o, Q_td):

    Q_td_new = Q_td.copy()
    Q_td_new = (1. - alpha_u) * Q_td_new + beta_u * (o - offset_u) # update unchosen actions
    Q_td_new[c] = (1. - alpha_c) * Q_td[c] + beta_c * (o - offset_c) # update chosen action

    return Q_td_new


@jit(nopython=True)
def _MFL_session_likelihood_core(alpha_c, offset_c, beta_c, alpha_u, offset_u, beta_u, iTemp, choices, second_steps, outcomes, Q_td, scores, choice_probs, n_trials, target):
    trial_log_likelihood = np.zeros(n_trials)
    for trial in range(n_trials):
        c, s, o = choices[trial], second_steps[trial], outcomes[trial]
        if target is None:
            trial_log_likelihood[trial] = _compute_loglik(choice_probs[trial], c)
        else:
            trial_log_likelihood[trial] = _compute_loglik(choice_probs[trial], target[trial])
        Q_td[trial + 1] = _MFL_step_core_variables(alpha_c, offset_c, beta_c, alpha_u, offset_u, beta_u, c, s, o, Q_td[trial])
        scores[trial + 1], choice_probs[trial + 1] = _step_other_variables(iTemp, Q_td[trial + 1])
    return trial_log_likelihood, Q_td, scores, choice_probs

class Model_free_learn(TwoStepModelCoreCSO):
    def __init__(self, n_actions=3, alpha_beta_sep=False):
        super().__init__()
        self.name = 'Model-free learn all'
        if alpha_beta_sep:
            self.param_names = ['alpha_c', 'offset_c', 'beta_c', 'alpha_u', 'offset_u', 'beta_u', 'iTemp']
            self.param_ranges = ['unit','unit','unc','unit','unit','unc', 'pos']
            self.params = [0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 5.]
            self.n_params = 7
        else:
            self.param_names = ['alpha_c', 'offset_c', 'alpha_u', 'offset_u', 'iTemp']
            self.param_ranges = ['unit','unit','unit','unit', 'pos']
            self.params = [0.5, 0.5, 0.5, 0.5, 5.]
            self.n_params = 5
            self.beta_c = self.params[0] # same as alpha_c
            self.beta_u = -self.params[2] # same as -alpha_u
        self.alpha_beta_sep = alpha_beta_sep
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
        if self.alpha_beta_sep:
            alpha_c, offset_c, beta_c, alpha_u, offset_u, beta_u, iTemp = params
        else:
            alpha_c, offset_c, alpha_u, offset_u, iTemp = params
            beta_c = self.beta_c
            beta_u = self.beta_u
        self.wm['Q_td'] = _MFL_step_core_variables(alpha_c, offset_c, beta_c, alpha_u, offset_u, beta_u, c, s, o, self.wm['Q_td'])

    def _step_other_variables(self, params):
        if self.alpha_beta_sep:
            alpha_c, offset_c, beta_c, alpha_u, offset_u, beta_u, iTemp = params
        else:
            alpha_c, offset_c, alpha_u, offset_u, iTemp = params
            beta_c = self.beta_c
            beta_u = self.beta_u
        self.wm['scores'], self.wm['choice_probs'] = _step_other_variables(iTemp, self.wm['Q_td'])

    def _session_likelihood_core(self, session, params, DVs):
        if self.alpha_beta_sep:
            alpha_c, offset_c, beta_c, alpha_u, offset_u, beta_u, iTemp = params
        else:
            alpha_c, offset_c, alpha_u, offset_u, iTemp = params
            beta_c = self.beta_c
            beta_u = self.beta_u
        assert np.all(session['choices'] == session['second_steps']), "MFD assumes that the first step action is the same as the second step state."
        target = session['target'] if 'target' in session else None
        DVs['trial_log_likelihood'], DVs['Q_td'], DVs['scores'], DVs['choice_probs'] = _MFL_session_likelihood_core(
            alpha_c, offset_c, beta_c, alpha_u, offset_u, beta_u, iTemp, session['choices'], session['second_steps'], session['outcomes'],
            DVs['Q_td'], DVs['scores'], DVs['choice_probs'], session['n_trials'], target)
        return DVs