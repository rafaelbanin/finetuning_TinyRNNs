"""Original model free model (single state) for the original two-step task + different utility for reward.
"""
import numpy as np
from numba import jit
from random import random, randint
from .core import TwoStepModelCoreOri, _compute_loglik, _step_other_variables

@jit(nopython=True)
def _OMFsr_step_core_variables(alpha1, alpha2, beta1, beta2, no_rew_1, no_rew_2, other_state_rew,
                             p_transit,
                             c1, s2, c2, o,
                             Q_stage1_net, Q_stage2_state1, Q_stage2_state2,
                             ):
    # alpha1: learning rate for first step action.
    # alpha2: learning rate for second step action.
    # beta1: inverse temperature for first step action.
    # beta2: inverse temperature for second step action.
    # no_rew_1: utility for first step action when no reward is given.
    # no_rew_2: utility for second step action when no reward is given.
    # other_state_rew: utility for second step action when reward is given in the other state.
    # p_transit: probability of transition.

    # assume all invalid trials are replaced.
    assert c1 == 0 or c1 == 1
    assert s2 == 0 or s2 == 1
    assert c2 == 0 or c2 == 1
    nc1 = 1 - c1  # Not chosen first step action.
    ns2 = 1 - s2  # Not reached second step state.
    nc2 = 1 - c2  # Not chosen second step action.

    Q_stage1_net_new = Q_stage1_net.copy()
    Q_stage2_state1_new = Q_stage2_state1.copy()
    Q_stage2_state2_new = Q_stage2_state2.copy()

    if o == 0: # no reward
        Q_stage1_net_new[c1] = (1 - alpha1) * Q_stage1_net[c1] + alpha1 * no_rew_1
    else: # reward
        Q_stage1_net_new[c1] = (1 - alpha1) * Q_stage1_net[c1] + alpha1 * 1
    Q_stage1_net_new[nc1] = - Q_stage1_net_new[c1] # anti-correlated update

    if s2 == 0 and o == 1: # stage 2 state 1, reward
        Q_stage2_state1_new[c2] = (1 - alpha2) * Q_stage2_state1[c2] + alpha2 * 1
        Q_stage2_state2_new[c2] = (1 - alpha2) * Q_stage2_state2[c2] + alpha2 * other_state_rew
    elif s2 == 0 and o == 0: # stage 2 state 1, no reward
        Q_stage2_state1_new[c2] = (1 - alpha2) * Q_stage2_state1[c2] - alpha2 * no_rew_2
        Q_stage2_state2_new[c2] = (1 - alpha2) * Q_stage2_state2[c2] - alpha2 * other_state_rew
    elif s2 == 1 and o == 1: # stage 2 state 2, reward
        Q_stage2_state1_new[c2] = (1 - alpha2) * Q_stage2_state1[c2] + alpha2 * other_state_rew
        Q_stage2_state2_new[c2] = (1 - alpha2) * Q_stage2_state2[c2] + alpha2 * 1
    elif s2 == 1 and o == 0: # stage 2 state 2, no reward
        Q_stage2_state1_new[c2] = (1 - alpha2) * Q_stage2_state1[c2] - alpha2 * other_state_rew
        Q_stage2_state2_new[c2] = (1 - alpha2) * Q_stage2_state2[c2] - alpha2 * no_rew_2
    else:
        raise ValueError
    Q_stage2_state1_new[nc2] = - Q_stage2_state1_new[c2] # anti-correlated update
    Q_stage2_state2_new[nc2] = - Q_stage2_state2_new[c2] # anti-correlated update

    return Q_stage1_net_new, Q_stage2_state1_new, Q_stage2_state2_new


@jit(nopython=True)
def _OMFsr_session_likelihood_core(alpha1, alpha2, beta1, beta2, no_rew_1, no_rew_2, other_state_rew,
                             p_transit,
                             first_choices, second_states, second_choices, outcomes, mask,
                             Q_stage1_net, Q_stage2_state1, Q_stage2_state2,
                             first_scores, first_choice_probs,
                             second_state1_scores, second_state1_choice_probs,
                             second_state2_scores, second_state2_choice_probs,
                             n_trials, target):
    trial_log_likelihood = np.zeros((n_trials, 2)) # first and second choices.
    trial_is_correct = -np.ones((n_trials, 2))
    for trial in range(n_trials):
        c1, s2, c2, o = first_choices[trial], second_states[trial], second_choices[trial], outcomes[trial]
        if target is None:
            trial_log_likelihood[trial, 0] = _compute_loglik(first_choice_probs[trial], c1)
            trial_is_correct[trial, 0] = (first_choice_probs[trial, c1] > 0.5) * 1
        else:
            prob = target[trial * 2][:2]
            assert np.isclose(np.sum(prob), 1)
            trial_log_likelihood[trial, 0] = _compute_loglik(first_choice_probs[trial], prob)
        if s2 == 0:
            if target is None:
                trial_log_likelihood[trial, 1] = _compute_loglik(second_state1_choice_probs[trial], c2)
                trial_is_correct[trial, 1] = (second_state1_choice_probs[trial, c2] > 0.5) * 1 # Warning: what about == 0.5?
            else:
                prob = target[trial * 2 + 1][2:4]
                assert np.isclose(np.sum(prob), 1)
                trial_log_likelihood[trial, 1] = _compute_loglik(second_state1_choice_probs[trial], prob)
        else:
            if target is None:
                trial_log_likelihood[trial, 1] = _compute_loglik(second_state2_choice_probs[trial], c2)
                trial_is_correct[trial, 1] = (second_state2_choice_probs[trial, c2] > 0.5) * 1
            else:
                prob = target[trial * 2 + 1][4:6]
                assert np.isclose(np.sum(prob), 1)
                trial_log_likelihood[trial, 1] = _compute_loglik(second_state2_choice_probs[trial], prob)
        Q_stage1_net[trial + 1], Q_stage2_state1[trial + 1], Q_stage2_state2[trial + 1] = \
            _OMFsr_step_core_variables(
            alpha1, alpha2, beta1, beta2, no_rew_1, no_rew_2, other_state_rew,
            p_transit,
            c1, s2, c2, o,
            Q_stage1_net[trial], Q_stage2_state1[trial], Q_stage2_state2[trial],
        )
        first_scores[trial + 1], first_choice_probs[trial + 1] = _step_other_variables(beta1, Q_stage1_net[trial + 1])
        second_state1_scores[trial + 1], second_state1_choice_probs[trial + 1] = _step_other_variables(beta2, Q_stage2_state1[trial + 1])
        second_state2_scores[trial + 1], second_state2_choice_probs[trial + 1] = _step_other_variables(beta2, Q_stage2_state2[trial + 1])
    trial_log_likelihood = trial_log_likelihood.flatten() * mask.flatten()
    trial_is_correct = trial_is_correct.flatten() * mask.flatten()
    return trial_log_likelihood, Q_stage1_net, Q_stage2_state1, Q_stage2_state2, \
             first_scores, first_choice_probs, second_state1_scores, second_state1_choice_probs, second_state2_scores, second_state2_choice_probs,trial_is_correct

class Origin_model_free_symm_rew(TwoStepModelCoreOri):
    def __init__(self, p_transit=0.7):
        super().__init__()
        self.name = 'Original model free symm rew'

        self.param_names =  ['alpha1', 'alpha2', 'beta1', 'beta2',
                             'no_rew_1', # no reward for first stage
                             'no_rew_2', # no reward for second stage
                             'other_state_rew', # reward when selecting other state
                             ]
        self.params = [0.4, 0.4, 5., 5., 0.5, 0.5, 0.5]
        self.param_ranges = ['unit', 'unit', 'pos', 'pos', 'unit', 'unit', 'unit']
        self.n_params = 7

        self.p_transit = p_transit
        self.state_vars = ['Q_stage1_net','Q_stage2_state1','Q_stage2_state2']

    def _init_core_variables(self, wm, params):
        if wm is None:
            self.wm = {
                'Q_stage1_net': np.zeros(2),
                'Q_stage2_state1': np.zeros(2),
                'Q_stage2_state2': np.zeros(2),
            }
        else:
            if 'h0' in wm:
                raise NotImplementedError
            else:
                self.wm = wm

    def _step_core_variables(self, trial_event, params):
        (c1, s2, c2, o) = trial_event
        alpha1, alpha2, beta1, beta2, no_rew_1, no_rew_2, other_state_rew = params
        self.wm['Q_stage1_net'], self.wm['Q_stage2_state1'], self.wm['Q_stage2_state2'] = _OMFsr_step_core_variables(
            alpha1, alpha2, beta1, beta2, no_rew_1, no_rew_2, other_state_rew,
            self.p_transit,
            c1, s2, c2, o,
            self.wm['Q_stage1_net'], self.wm['Q_stage2_state1'], self.wm['Q_stage2_state2']
        )

    def _step_other_variables(self, params):
        alpha1, alpha2, beta1, beta2, no_rew_1, no_rew_2, other_state_rew = params
        self.wm['first_scores'], self.wm['first_choice_probs'] = _step_other_variables(beta1, self.wm['Q_stage1_net'])
        self.wm['second_state1_scores'], self.wm['second_state1_choice_probs'] = _step_other_variables(beta2, self.wm['Q_stage2_state1'])
        self.wm['second_state2_scores'], self.wm['second_state2_choice_probs'] = _step_other_variables(beta2, self.wm['Q_stage2_state2'])

    def _session_likelihood_core(self, session, params, DVs):
        alpha1, alpha2, beta1, beta2, no_rew_1, no_rew_2, other_state_rew = params
        target = session['target'] if 'target' in session else None
        DVs['trial_log_likelihood'], DVs['Q_stage1_net'], DVs['Q_stage2_state1'], DVs['Q_stage2_state2'], \
            DVs['first_scores'], DVs['first_choice_probs'], DVs['second_state1_scores'], DVs['second_state1_choice_probs'], DVs['second_state2_scores'], DVs['second_state2_choice_probs'],DVs['trial_is_correct'] = _OMFsr_session_likelihood_core(
            alpha1, alpha2, beta1, beta2, no_rew_1, no_rew_2, other_state_rew,
            self.p_transit,
            session['first_choices'], session['second_states'], session['second_choices'], session['outcomes'], session['mask'],
            DVs['Q_stage1_net'], DVs['Q_stage2_state1'], DVs['Q_stage2_state2'],
            DVs['first_scores'], DVs['first_choice_probs'],
            DVs['second_state1_scores'], DVs['second_state1_choice_probs'],
            DVs['second_state2_scores'], DVs['second_state2_choice_probs'],
            session['n_trials'], target)
        scores = np.zeros([2 * session['n_trials'] + 2, 6])
        # include the last unseen trial
        for stage_in_trial in range(2):
            scores[stage_in_trial::2, 0:2] = DVs['first_scores']
            scores[stage_in_trial::2, 2:4] = DVs['second_state1_scores']
            scores[stage_in_trial::2, 4:6] = DVs['second_state2_scores']

        choice_probs = np.zeros([2 * session['n_trials'] + 1, 6])
        choice_probs[0::2, 0:2] = DVs['first_choice_probs'] # include the last unseen trial
        choice_probs[1::2, 2:4] = DVs['second_state1_choice_probs'][:-1]  # exclude the last unseen trial
        choice_probs[1::2, 4:6] = DVs['second_state2_choice_probs'][:-1]  # exclude the last unseen trial
        DVs['scores'] = scores
        DVs['choice_probs'] = choice_probs
        return DVs