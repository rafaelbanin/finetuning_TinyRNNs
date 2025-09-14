"""Reward as cue model, model-free. """
import numpy as np
from numba import jit
from random import random, randint
from .core import TwoStepModelCoreCSO, _compute_loglik, _step_other_variables

@jit(nopython=True)
def _RC_step_core_variables(alpha, c, s, o, Q_td, Q, ps, po):
    nc = 1 - c  # Not chosen first step action.
    ns = 1 - s  # Not reached second step state.

    Q_td_new = Q_td.copy()
    Q_new = Q.copy()
    ps = int(ps[0])
    po = int(po[0])
    Q_td_new[c, ps, po] = (1. - alpha) * Q_td[c, ps, po] + alpha * o
    Q_new[0] = Q_td_new[0, s, o]
    Q_new[1] = Q_td_new[1, s, o]

    ps, po = np.array([s]), np.array([o])

    return Q_td_new, Q_new, ps, po


@jit(nopython=True)
def _RC_session_likelihood_core(alpha, iTemp, choices, second_steps, outcomes, Q_td, Q, ps, po, scores, choice_probs, n_trials):
    trial_log_likelihood = np.zeros(n_trials)
    for trial in range(n_trials):
        c, s, o = choices[trial], second_steps[trial], outcomes[trial]
        trial_log_likelihood[trial] = _compute_loglik(choice_probs[trial], c)
        Q_td[trial + 1], Q[trial + 1], ps[trial + 1], po[trial + 1] = _RC_step_core_variables(alpha, c, s, o, Q_td[trial], Q[trial], ps[trial], po[trial])
        scores[trial + 1], choice_probs[trial + 1] = _step_other_variables(iTemp, Q[trial + 1])
    return trial_log_likelihood, Q_td, Q, ps, po, scores, choice_probs

class Reward_as_cue(TwoStepModelCoreCSO):
    def __init__(self):
        super().__init__()
        self.name = 'Reward as cue'
        self.param_names = ['alpha', 'iTemp']
        self.params = [ 0.05   ,  6.   ]
        self.param_ranges = ['unit', 'pos']
        self.n_params = 2
        self.state_vars = ['Q_td']

    def _init_core_variables(self, wm, params):
        if wm is None:
            self.wm = {
                'Q_td': np.zeros([2, 2, 2]),
                'Q': np.zeros(2),
                'ps': np.zeros(1),
                'po': np.zeros(1),
            }
        else:
            if 'h0' in wm:
                raise NotImplementedError
            else:
                self.wm = wm

    def _step_core_variables(self, trial_event, params):
        (c, s, o) = trial_event
        alpha, iTemp = params
        self.wm['Q_td'], self.wm['Q'], self.wm['ps'], self.wm['po'] = _RC_step_core_variables(alpha, c, s, o, self.wm['Q_td'], self.wm['Q'], self.wm['ps'], self.wm['po'] )

    def _step_other_variables(self, params):
        alpha, iTemp = params
        self.wm['scores'], self.wm['choice_probs'] = _step_other_variables(iTemp, self.wm['Q'])

    def _session_likelihood_core(self, session, params, DVs):
        alpha, iTemp = params
        DVs['trial_log_likelihood'], DVs['Q_td'], DVs['Q'], DVs['ps'], DVs['po'], DVs['scores'], DVs['choice_probs'] = _RC_session_likelihood_core(
            alpha, iTemp, session['choices'], session['second_steps'], session['outcomes'],
            DVs['Q_td'], DVs['Q'], DVs['ps'], DVs['po'], DVs['scores'], DVs['choice_probs'], session['n_trials'])
        return DVs