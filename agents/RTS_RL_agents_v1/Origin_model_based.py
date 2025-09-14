"""Original model based model for the original two-step task.
"""
import numpy as np
from numba import jit
from random import random, randint
from .core import TwoStepModelCoreOri, _compute_loglik, _step_other_variables

@jit(nopython=True)
def _OMB_step_core_variables(alpha1, alpha2, beta1, beta2, lbd, pers, w,
                             p_transit,
                             c1, s2, c2, o,
                             Q_stage1_MF, Q_stage1_MB, Q_stage1_net, Q_stage2_state1, Q_stage2_state2, pc1,
                             ):
    # alpha1: learning rate for first step action.
    # alpha2: learning rate for second step action.
    # beta1: inverse temperature for first step action.
    # beta2: inverse temperature for second step action.
    # lbd: sarsa lambda.
    # pers: perseveration parameter.
    # w: weight of MB.
    # p_transit: probability of transition.

    # assume all invalid trials are replaced.
    assert c1 == 0 or c1 == 1
    assert s2 == 0 or s2 == 1
    assert c2 == 0 or c2 == 1
    nc1 = 1 - c1  # Not chosen first step action.
    ns2 = 1 - s2  # Not reached second step state.
    nc2 = 1 - c2  # Not chosen second step action.

    Q_stage1_MF_new = Q_stage1_MF.copy()
    Q_stage2_state1_new = Q_stage2_state1.copy()
    Q_stage2_state2_new = Q_stage2_state2.copy()
    pc1 = int(pc1[0]) # first-stage choice in previous trial
    choice_pers = np.zeros(2)
    choice_pers[pc1] = pers

    if s2 == 0: # stage 2 state 1
        Q_stage1_MF_new[c1] = (1 - alpha1) * Q_stage1_MF[c1] + alpha1 * (
                    (1 - lbd) * Q_stage2_state1[c2] + lbd * o)
        Q_stage2_state1_new[c2] = (1 - alpha2) * Q_stage2_state1[c2] + alpha2 * o
    else:
        Q_stage1_MF_new[c1] = (1 - alpha1) * Q_stage1_MF[c1] + alpha1 * (
                    (1 - lbd) * Q_stage2_state2[c2] + lbd * o)
        Q_stage2_state2_new[c2] = (1 - alpha2) * Q_stage2_state2[c2] + alpha2 * o
    V_stage2 = np.array([np.max(Q_stage2_state1_new), np.max(Q_stage2_state2_new)])
    Q_stage1_MB_new = p_transit * V_stage2 + (1 - p_transit) * V_stage2[::-1]
    Q_stage1_net_new = (1 - w) * Q_stage1_MF_new + w * Q_stage1_MB_new + choice_pers

    pc1 = np.array([c1])
    return Q_stage1_MF_new, Q_stage1_MB_new, Q_stage1_net_new, Q_stage2_state1_new, Q_stage2_state2_new, pc1


@jit(nopython=True)
def _OMB_session_likelihood_core(alpha1, alpha2, beta1, beta2, lbd, pers, w,
                             p_transit,
                             first_choices, second_states, second_choices, outcomes, mask,
                             Q_stage1_MF, Q_stage1_MB, Q_stage1_net, Q_stage2_state1, Q_stage2_state2, prev_first_choices,
                             first_scores, first_choice_probs,
                             second_state1_scores, second_state1_choice_probs,
                             second_state2_scores, second_state2_choice_probs,
                             n_trials,target):
    trial_log_likelihood = np.zeros((n_trials, 2)) # first and second choices.
    trial_is_correct = -np.ones((n_trials, 2), dtype=np.float32) # first and second choices.
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
                trial_is_correct[trial, 1] = (second_state1_choice_probs[trial, c2] > 0.5) * 1
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
        Q_stage1_MF[trial + 1], Q_stage1_MB[trial + 1], Q_stage1_net[trial + 1], Q_stage2_state1[trial + 1], Q_stage2_state2[trial + 1], prev_first_choices[trial + 1] = \
            _OMB_step_core_variables(
            alpha1, alpha2, beta1, beta2, lbd, pers, w,
            p_transit,
            c1, s2, c2, o,
            Q_stage1_MF[trial], Q_stage1_MB[trial], Q_stage1_net[trial], Q_stage2_state1[trial], Q_stage2_state2[trial], prev_first_choices[trial],
        )
        first_scores[trial + 1], first_choice_probs[trial + 1] = _step_other_variables(beta1, Q_stage1_net[trial + 1])
        second_state1_scores[trial + 1], second_state1_choice_probs[trial + 1] = _step_other_variables(beta2, Q_stage2_state1[trial + 1])
        second_state2_scores[trial + 1], second_state2_choice_probs[trial + 1] = _step_other_variables(beta2, Q_stage2_state2[trial + 1])
    trial_log_likelihood = trial_log_likelihood.flatten() * mask.flatten()
    trial_is_correct = trial_is_correct.flatten() * mask.flatten()
    return trial_log_likelihood, Q_stage1_MF, Q_stage1_MB, Q_stage1_net, Q_stage2_state1, Q_stage2_state2, prev_first_choices,\
             first_scores, first_choice_probs, second_state1_scores, second_state1_choice_probs, second_state2_scores, second_state2_choice_probs, trial_is_correct

class Origin_model_based(TwoStepModelCoreOri):
    def __init__(self, p_transit=0.7, model_type='MF'):
        super().__init__()
        assert model_type in ['MF', 'MB', 'MX']
        self.model_type = model_type
        self.name = 'Original model based ' + model_type
        if model_type == 'MF':
            self.param_names =  ['alpha1', 'alpha2', 'beta1', 'beta2', 'lbd', 'pers']
            self.w = 0
            self.params = [0.4, 0.4, 5., 5., 0.5, 0.]
            self.param_ranges = ['unit', 'unit', 'pos', 'pos', 'unit', 'unc']
            self.n_params = 6
        elif model_type == 'MB':
            self.param_names = ['alpha2', 'beta1', 'beta2', 'pers']
            self.alpha1 = 0
            self.lbd = 0
            self.w = 1
            self.params = [0.4, 5., 5., 0.]
            self.param_ranges = ['unit', 'pos', 'pos', 'unc']
            self.n_params = 4
        elif model_type == 'MX':
            self.param_names = ['alpha1', 'alpha2', 'beta1', 'beta2', 'lbd', 'pers', 'w']
            self.params = [0.4, 0.4, 5., 5., 0.5, 0., 0.5]
            self.param_ranges = ['unit', 'unit', 'pos', 'pos', 'unit', 'unc', 'unit']
            self.n_params = 7
        self.p_transit = p_transit
        self.state_vars = ['Q_stage1_net','Q_stage2_state1','Q_stage2_state2']

    def _init_core_variables(self, wm, params):
        if wm is None:
            self.wm = {
                'Q_stage1_MF': np.zeros(2),
                'Q_stage1_MB': np.zeros(2),
                'Q_stage1_net': np.zeros(2),
                'Q_stage2_state1': np.zeros(2),
                'Q_stage2_state2': np.zeros(2),
                'prev_first_choices': np.zeros(1),
            }
        else:
            if 'h0' in wm:
                raise NotImplementedError
            else:
                self.wm = wm

    def _step_core_variables(self, trial_event, params):
        (c1, s2, c2, o) = trial_event
        if self.model_type == 'MF':
            alpha1, alpha2, beta1, beta2, lbd, pers = params
            w = self.w
        elif self.model_type == 'MB':
            alpha2, beta1, beta2, pers = params
            alpha1 = self.alpha1
            lbd = self.lbd
            w = self.w
        elif self.model_type == 'MX':
            alpha1, alpha2, beta1, beta2, lbd, pers, w = params
        else:
            raise NotImplementedError
        self.wm['Q_stage1_MF'], self.wm['Q_stage1_MB'], self.wm['Q_stage1_net'], self.wm['Q_stage2_state1'], self.wm['Q_stage2_state2'], self.wm['prev_first_choices'] = _OMB_step_core_variables(
            alpha1, alpha2, beta1, beta2, lbd, pers, w,
            self.p_transit,
            c1, s2, c2, o,
            self.wm['Q_stage1_MF'], self.wm['Q_stage1_MB'], self.wm['Q_stage1_net'], self.wm['Q_stage2_state1'], self.wm['Q_stage2_state2'], self.wm['prev_first_choices'],
        )

    def _step_other_variables(self, params):
        if self.model_type == 'MF':
            alpha1, alpha2, beta1, beta2, lbd, pers = params
        elif self.model_type == 'MB':
            alpha2, beta1, beta2, pers = params
        elif self.model_type == 'MX':
            alpha1, alpha2, beta1, beta2, lbd, pers, w = params
        else:
            raise NotImplementedError
        self.wm['first_scores'], self.wm['first_choice_probs'] = _step_other_variables(beta1, self.wm['Q_stage1_net'])
        self.wm['second_state1_scores'], self.wm['second_state1_choice_probs'] = _step_other_variables(beta2, self.wm['Q_stage2_state1'])
        self.wm['second_state2_scores'], self.wm['second_state2_choice_probs'] = _step_other_variables(beta2, self.wm['Q_stage2_state2'])

    def _session_likelihood_core(self, session, params, DVs):
        if self.model_type == 'MF':
            alpha1, alpha2, beta1, beta2, lbd, pers = params
            w = self.w
        elif self.model_type == 'MB':
            alpha2, beta1, beta2, pers = params
            alpha1 = self.alpha1
            lbd = self.lbd
            w = self.w
        elif self.model_type == 'MX':
            alpha1, alpha2, beta1, beta2, lbd, pers, w = params
        else:
            raise NotImplementedError
        target = session['target'] if 'target' in session else None
        DVs['trial_log_likelihood'], DVs['Q_stage1_MF'], DVs['Q_stage1_MB'], DVs['Q_stage1_net'], DVs['Q_stage2_state1'], DVs['Q_stage2_state2'], DVs['prev_first_choices'],\
            DVs['first_scores'], DVs['first_choice_probs'], DVs['second_state1_scores'], DVs['second_state1_choice_probs'], DVs['second_state2_scores'], DVs['second_state2_choice_probs'],DVs['trial_is_correct'] = _OMB_session_likelihood_core(
            alpha1, alpha2, beta1, beta2, lbd, pers, w,
            self.p_transit,
            session['first_choices'], session['second_states'], session['second_choices'], session['outcomes'], session['mask'],
            DVs['Q_stage1_MF'], DVs['Q_stage1_MB'], DVs['Q_stage1_net'], DVs['Q_stage2_state1'], DVs['Q_stage2_state2'], DVs['prev_first_choices'],
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