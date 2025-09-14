"""Latent state model with softmax action selection. The LS model with additional bias. """
import numpy as np
from numba import jit
from random import random, randint
from .core import TwoStepModelCoreCSO, _compute_loglik, _step_other_variables


@jit(nopython=True)
def _LS_step_core_variables(p_r, bias1, bias2, good_prob, c, s, o, p_1):
    nc = 1 - c  # Not chosen first step action.
    ns = 1 - s  # Not reached second step state.

    p_o_1 = np.array([[good_prob, 1 - good_prob],  # Probability of observed outcome given world in state 1.
                      [1 - good_prob, good_prob]])  # Indices:  p_o_1[second_step, outcome]
    p_o_0 = 1 - p_o_1  # Probability of observed outcome given world in state 0.
    p_o_0[0] *= bias1
    p_o_0[1] *= bias2

    # Bayesian update of state probabilities given observed outcome.
    p_1_new = p_o_1[s, o] * p_1 / (p_o_1[s, o] * p_1 + p_o_0[s, o] * (1 - p_1))
    # Update of state probabilities due to possibility of block reversal.
    p_1_new = (1 - p_r) * p_1_new + p_r * (1 - p_1_new)

    return p_1_new


@jit(nopython=True)
def _LS_session_likelihood_core(p_r, bias1, bias2, iTemp, good_prob, choices, second_steps, outcomes, p_1, scores, choice_probs,
                                n_trials):
    trial_log_likelihood = np.zeros(n_trials)
    for trial in range(n_trials):
        c, s, o = choices[trial], second_steps[trial], outcomes[trial]
        trial_log_likelihood[trial] = _compute_loglik(choice_probs[trial], c)
        p_1[trial + 1] = _LS_step_core_variables(p_r, bias1, bias2, good_prob, c, s, o, p_1[trial])
        scores[trial + 1], choice_probs[trial + 1] = _step_other_variables(iTemp, np.array([1 - p_1[trial + 1], p_1[trial + 1]]))
    return trial_log_likelihood, p_1, scores, choice_probs


class Latent_state_softmax_bias(TwoStepModelCoreCSO):
    """
    Latent state model with softmax action selection. The LS model with additional bias.
    The state variable is p_1, the probability of the task being in latent state 1.
    """
    def __init__(self, good_prob=0.8):
        super().__init__()
        self.name = 'Latent state softmax biased'
        self.param_names = ['p_r', 'bias1', 'bias2', 'iTemp']
        self.params = [0.1, 1, 1, 5.]
        self.param_ranges = ['half', 'pos','pos', 'pos']
        self.n_params = 4
        self.good_prob = good_prob
        self.state_vars = ['p_1']

    def _init_core_variables(self, wm, params):
        if wm is None:
            self.wm = {
                'p_1': np.array(0.5),
            }
        else:
            if 'h0' in wm:
                raise NotImplementedError
            else:
                self.wm = wm

    def _step_core_variables(self, trial_event, params):
        (c, s, o) = trial_event
        p_r, bias1, bias2, iTemp = params
        self.wm['p_1'] = _LS_step_core_variables(p_r, bias1, bias2, self.good_prob, c, s, o, self.wm['p_1'])

    def _step_other_variables(self, params):
        p_r, bias1, bias2, iTemp = params
        self.wm['scores'], self.wm['choice_probs'] = _step_other_variables(iTemp, np.array([1 - self.wm['p_1'], self.wm['p_1']]))

    def _session_likelihood_core(self, session, params, DVs):
        p_r, bias1, bias2, iTemp = params
        DVs['trial_log_likelihood'], DVs['p_1'], DVs['scores'], DVs['choice_probs'] = _LS_session_likelihood_core(
            p_r, bias1, bias2, iTemp, self.good_prob, session['choices'], session['second_steps'], session['outcomes'],
            DVs['p_1'], DVs['scores'], DVs['choice_probs'], session['n_trials'])
        return DVs