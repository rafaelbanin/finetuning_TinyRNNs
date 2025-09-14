"""Customized RNN agents."""
from .BaseAgent import BaseAgent
from .network_models import RNNnet, set_seed
from copy import deepcopy
import torch
import os
import json
import joblib
from path_settings import *
import numpy as np
import torch.nn as nn
from random import random

class RNNAgent(BaseAgent):
    """All types of RNN agents.

    Attributes:
        config:
        model: the RNN network, implemented in PyTorch way.
        behav_loss_function: a loss function for behavior prediction
        num_params: total number of parameters in the model

    """

    def __init__(self, config=None, compile=False):
        super().__init__()
        self.config = config
        assert config['rnn_type'] in ['GRU', 'SGRU', 'MIGRU', 'PNR1', 'PNR2','LR','MLR']
        other_config = deepcopy(config)
        [other_config.pop(key) for key in ['input_dim', 'hidden_dim', 'output_dim']]
        set_seed(config['seed'])
        self.rnn_type = config['rnn_type']
        self.model = RNNnet(config['input_dim'], config['hidden_dim'], config['output_dim'], **other_config).double()
        self.model.share_memory() # required for multiprocessing
        if compile:
            self.model = torch.compile(self.model)
        self.num_params = sum(param.numel() for param in self.model.parameters())
        self.behav_loss_function = nn.CrossEntropyLoss(reduction='none')
        if 'device' in config:
            self.model.to(config['device'])

    @property
    def params(self):
        return self.model.parameters()

    def load(self, model_path=None, strict=False,mode='train'):
        """Load model parameters from disk."""
        if model_path is None:
            model_path = self.config['model_path']
        state_dict = torch.load(MODEL_SAVE_PATH / model_path / 'model.ckpt', map_location=torch.device(self.config['device']))
        self.model.load_state_dict(state_dict, strict=strict)
        if mode=='eval':
            self.model.eval()
        else:
            assert mode=='train'
            self.model.train()

    def forward(self, input, h0=None, standard_output=False, pointwise_loss=False, demask=True):
        """Process a batch of input.

        Args:
            input: (1) Tensor shape: seq_len, batch_size, input_dim
                (2) or a dict containing 'input' and 'mask', each of which is a Tensor
            h0: initial hidden state
            standard_output: if True, returns a dict of list
                each list includes the output of each episode
        Returns:
            a dict of internal states and outputs
        """
        if isinstance(input, dict):
            input_ = input['input']
        else:
            input_ = input
        if h0 is not None:
            assert len(h0.shape) == 1
            h0 = h0[None, None, :]
            if isinstance(h0, np.ndarray):
                h0 = torch.from_numpy(h0).to(device=input_.device)
        scores, rnn_out = self.model(input_, get_rnnout=True, h0=h0)
        behav_loss = None
        total_corrected_trials = None
        if pointwise_loss: # if we want to return trial-wise loss; otherwise, behav_loss is None
            model_pass = self._eval_1step(input['input'], input['target'], input['mask'], h0=h0)
            scores = model_pass['output'] # notice that scores could be affected by target_subtractive_mask
            behav_loss = model_pass['behav_loss_total'].detach().cpu().numpy() # shape: seq_len, batch_size
            total_corrected_trials = model_pass['total_corrected_trials'].detach().cpu().numpy() # shape: seq_len, batch_size
        # now the scores and rnn_out are in tensor format and masked
        if standard_output: # if we want to return the list of blocks in numpy format
            # if demask is True, the scores and rnn_out are demasked; otherwise, they are unchanged
            # this procedure assumes that the mask is 0 at the end of each episode, for the padding
            # the nonvalid values in scores and rnn_out are ignored and unchanged
            assert 'mask' in input
            output_layer_num = self.model.output_layer_num # if >1, scores is a list of tensors
            if isinstance(input['mask'], tuple):
                assert len(input['mask']) == 2
                mask = input['mask'][0].detach().cpu().numpy()
                other_mask = input['mask'][1].detach().cpu().numpy()
            else:
                mask = input['mask'].detach().cpu().numpy() # shape: seq_len, batch_size
                other_mask = None
            padding_mask = mask.copy()
            scores_list = []
            rnn_out_list = []
            behav_loss_list = []
            total_corrected_trials_list = []
            mask_list = []
            scores = scores.detach().cpu().numpy() if output_layer_num == 1 else [x.detach().cpu().numpy() for x in scores]
            rnn_out = rnn_out.detach().cpu().numpy()
            for i in range(mask.shape[1]):
                if demask:
                    # make sure 0 in mask only appears at the end of each episode
                    if np.any(padding_mask[:, i] == 0):
                        first_zero_loc = np.where(padding_mask[:, i] == 0)[0][0]
                        if not np.all(padding_mask[first_zero_loc:, i] == 0):
                            print('Warning: there are non-zero values after 0 in mask, be careful when analyzing the results.')
                        last_one_loc = np.where(padding_mask[:, i] == 1)[0][-1]
                        # make padding mask 1 before the last_one_loc, and 0 after the last_one_loc
                        padding_mask[:last_one_loc, i] = 1 # now padding_mask might be different from the original mask
                    scores_list.append(
                        scores[:int(padding_mask[:, i].sum()) + 1, i, :] if output_layer_num == 1 else [x[:int(padding_mask[:, i].sum()) + 1, i, :] for x in scores]
                    )
                    rnn_out_list.append(rnn_out[:int(padding_mask[:, i].sum()) + 1, i, :])
                    if other_mask is None:
                        mask_list.append(mask[:int(padding_mask[:, i].sum()), i])
                    else:
                        mask_list.append((mask[:int(padding_mask[:, i].sum()), i], other_mask[:int(padding_mask[:, i].sum()), i]))

                    if pointwise_loss:
                        behav_loss_list.append(behav_loss[:int(padding_mask[:, i].sum()), i]) # score length is 1 longer than behav_loss
                        total_corrected_trials_list.append(total_corrected_trials[:int(padding_mask[:, i].sum()), i])
                else:
                    scores_list.append(scores[:, i, :]
                                        if output_layer_num == 1 else [x[:, i, :] for x in scores])
                    rnn_out_list.append(rnn_out[:, i, :])
                    if other_mask is None:
                        mask_list.append(mask[:, i])
                    else:
                        mask_list.append((mask[:, i], other_mask[:, i]))
                    if pointwise_loss:
                        behav_loss_list.append(behav_loss[:, i])
                        total_corrected_trials_list.append(total_corrected_trials[:, i])
            scores = scores_list
            rnn_out = rnn_out_list
            if pointwise_loss:
                behav_loss = behav_loss_list
                total_corrected_trials = total_corrected_trials_list
            return {'output': scores, 'internal': rnn_out, 'behav_loss': behav_loss, 'mask': mask_list, 'total_corrected_trials': total_corrected_trials}

        return {'output': scores, 'internal': rnn_out, 'behav_loss': behav_loss, 'total_corrected_trials': total_corrected_trials}

    def save(self, params=None, verbose=False):
        """Save config, model, and results."""
        model_path = self.config['model_path']
        os.makedirs(MODEL_SAVE_PATH / model_path, exist_ok=True)
        # save config
        self.save_config()

        # save model parameters
        if params is None: # current state dict is saved
            torch.save(self.model.state_dict(), MODEL_SAVE_PATH / model_path / 'model.ckpt')
        else:
            torch.save(params, MODEL_SAVE_PATH / model_path / 'model.ckpt')

        if verbose: print('Saved model at', MODEL_SAVE_PATH / model_path)

    def _eval_1step(self, input, target, mask, h0=None):
        """Return loss on (test/val) data set without gradient update."""
        with torch.no_grad():
            model_pass = self._compare_to_target(input, target, mask, h0=h0)
            # self._to_numpy(model_pass)
        return model_pass

    def _compare_to_target(self, input, target, mask, h0=None):
        """Compare model's output to target and compute losses.

        Args:
            input: shape (seq_len, batch_size, input_dim), 0 pading for shorter sequences
            target: shape (seq_len, batch_size) as class, or (seq_len, batch_size, output_dim) as prob., 0 pading for shorter sequences
            mask: shape (seq_len, batch_size), 1 for valid, 0 for invalid
                if tuple,
                    mask[0]: shape (seq_len, batch_size), 1 for valid, 0 for invalid
                    mask[1]: shape (seq_len, batch_size, output_size), large positive number for invalid, 0 for valid
            h0: initial hidden state of the model.
        Returns:
            model_pass: a dict containing all the information of this pass.
        """
        model_pass = self.forward(input, h0=h0)
        model_pass['input'] = input
        model_pass['target'] = target
        model_pass['mask'] = mask

        scores = model_pass['output'] # shape: seq_len, batch_size, output_dim, or a list of scores
        output_dim = self.config['output_dim']
        if isinstance(scores, torch.Tensor):
            multi_scores = [
                scores
            ] # compatible with single target; we want to make sure this list is updated when scores is updated
            multi_score_num = 1
        else:
            assert isinstance(scores, list), type(scores)
            multi_scores = scores
            multi_score_num = len(scores)
        rnn_out = model_pass['internal']
        if isinstance(target, torch.Tensor):
            multi_target = [
                ('action',1, target) # compatible with multiple targets
                # target_name, loss_proportion, target
            ]
            multi_target_num = 1
        else:
            assert isinstance(target, list)
            multi_target = target
            multi_target_num = len(target)

        proportion_sum = np.sum([x[1] for x in multi_target])
        assert proportion_sum == 1, f'loss proportion ({[x[1] for x in multi_target]}) should sum to 1'

        # use the shape of the first target to check the shape of the scores
        target_shape = multi_target[0][2].shape
        if self.config['output_h0']:
            for scores_idx in range(len(multi_scores)):
                scores = multi_scores[scores_idx]
                assert scores.shape[0] - 1 == target_shape[0], (scores.shape, target_shape)
                # because output_h0 is True, scores will have one more time dimension than target/input/mask
                scores = scores[:-1] # remove the last time dimension
                multi_scores[scores_idx] = scores
            rnn_out = rnn_out[:-1]
        else:
            for scores in multi_scores:
                assert scores.shape[0] == target_shape[0]

        if isinstance(mask, tuple):
            assert multi_score_num == 1, 'Not implemented for multiple scores and multiple masks'
            mask, target_subtractive_mask = mask
            # WARNING: some trained models may have a bug that the target_subtractive_mask is not subtracted from the scores
            # these models used the following line
            # scores = scores - target_subtractive_mask # mask out scores for invalid actions

            # the following line is the correct way to mask out scores
            for scores_idx in range(len(multi_scores)):
                scores = multi_scores[scores_idx] - target_subtractive_mask
                multi_scores[scores_idx] = scores
        else:
            target_subtractive_mask = None
        # Notice: mask out score and rnn_out here is not necessary, but no effect on the result
        for scores_idx in range(len(multi_scores)):
            scores = multi_scores[scores_idx]
            multi_scores[scores_idx] = scores * mask[..., None] # mask out one additional time step, scores.shape: trial_num, batch_size, output_dim
        rnn_out = rnn_out * mask[..., None]
        if multi_score_num == 1:
            model_pass['output'] = multi_scores[0]
        else:
            model_pass['output'] = multi_scores
        model_pass['internal'] = rnn_out
        assert multi_scores[0].shape[-1] == output_dim

        if multi_score_num == multi_target_num:
            # simplest case for one score and one target
            # or multiple scores and multiple targets (multiple target prediction)
            target_index_to_score_index = lambda idx: idx
        elif multi_score_num == 1 and multi_target_num > 1: # teacher target case
            target_index_to_score_index = lambda idx: 0
        else:
            raise ValueError('multi_score_num and multi_target_num not compatible.')
        for target_idx, (target_name, loss_proportion, target) in enumerate(multi_target):
            scores = multi_scores[target_index_to_score_index(target_idx)]
            scores_temp = scores.reshape([-1, scores.shape[-1]]) # shape: seq_len*batch_size, output_dim(or output_dim1)
            loss_correction = 1
            if len(target.shape) == 2: # (seq_len, batch_size) as class
                target_temp = target.flatten()
                loss_shape = target.shape # shape: seq_len, batch_size
            elif len(target.shape) == 3: # (seq_len, batch_size, output_dim) as prob.
                target_temp = target.reshape([-1, target.shape[-1]]) # shape: seq_len*batch_size, output_dim
                if 'distill_temp' in self.config:
                    distill_temp = self.config['distill_temp']
                else:
                    distill_temp = 1
                scores_temp = scores_temp / distill_temp
                loss_correction = distill_temp ** 2
                loss_shape = target.shape[:-1]
            else:
                raise ValueError('target should be 2 or 3 dimensional.')
            behav_loss_total = self.behav_loss_function(scores_temp, target_temp).reshape(loss_shape) # shape: seq_len, batch_size
            behav_loss_total = behav_loss_total * mask * loss_correction
            total_trial_num = torch.sum(mask)
            behav_loss = behav_loss_total.sum() / total_trial_num # only average over valid trials
            model_pass['behav_loss_' + target_name] = behav_loss
            model_pass['behav_loss_total_' + target_name] = behav_loss_total

            if 'behav_loss_total' not in model_pass: # first time
                model_pass['behav_loss_total'] = behav_loss_total * loss_proportion
                model_pass['behav_loss'] = behav_loss * loss_proportion
            else: # add to the previous loss
                model_pass['behav_loss_total'] += behav_loss_total * loss_proportion
                model_pass['behav_loss'] += behav_loss * loss_proportion
        model_pass['all_target_names'] = [x[0] for x in multi_target]
        model_pass['behav_mask_total'] = mask
        model_pass['total_trial_num'] = total_trial_num
        if len(target.shape) == 2:
            assert target.shape == mask.shape # shape: seq_len, batch_size
        else:
            # target shape: seq_len, batch_size, action_dim; due to teacher's target
            assert target.shape[:-1] == mask.shape, (target.shape, mask.shape)
            target = torch.argmax(target, dim=-1) # shape: seq_len, batch_size
        corrected_trials = (torch.argmax(multi_scores[0][:target.shape[0]], dim=-1) == target).float() * mask # shape: seq_len, batch_size
        model_pass['total_corrected_trials'] = corrected_trials

        return model_pass

    def simulate(self, task, n_trials, params=None, get_DVs=False, update_session=None, pack_trial_event=None, use_one_hot='false'):
        """Interact with task. Return session data and decision variables.  """
        assert params is None

        session = {'n_trials': n_trials} # 'choices', 'second_steps', 'outcomes'
        DVs = {
            'scores': np.zeros((n_trials + 1, self.config['output_dim'])),
            'rnn_out': np.zeros((n_trials + 1, self.config['hidden_dim'])),
            'choice_probs': np.zeros((n_trials + 1, self.config['output_dim'])),
        }  # each value is a numpy array of shape (n_trials + 1, ...), +1 for initial state

        task.reset(n_trials)
        # init_input is only used to elicit the initial rnn_out and scores before the first trial input
        # the exact value is not important, as long as it is a valid input
        init_input = torch.zeros(1, 1, self.config['input_dim'], dtype=torch.double,
                                 device=self.config['device']) # shape: seq_len=1, batch_size=1, input_dim
        init_input[0, 0, 0] = 1 # this line make it works also for SGRU
        scores, rnn_out = self.model(init_input, get_rnnout=True, h0=None)
        assert scores.shape[0] == 2 and scores.shape[1] == 1, scores.shape # scores.shape: seq_len, batch_size, output_dim

        DVs['rnn_out'][0] = rnn_out[0, 0, :].detach().cpu().numpy() # shape: hidden_dim
        DVs['scores'][0] = scores[0, 0, :].detach().cpu().numpy() # shape: output_dim
        DVs['choice_probs'][0] = scores[0, 0, :].softmax(dim=-1).detach().cpu().numpy() # shape: output_dim

        hn = rnn_out[0:1, 0:1, :] # shape: seq_len=1, batch_size=1, hidden_dim
        for trial in range(n_trials):
            # Generate the trial event.
            c = choose(DVs['choice_probs'][trial])
            obs = task.trial(c)  # usually (s, o)
            trial_event = pack_trial_event(c, obs, use_one_hot=use_one_hot)  # usually (c, s, o)
            trial_input = torch.from_numpy(trial_event)[None, None, :].double().to(self.config['device']) # shape: seq_len=1, batch_size=1, input_dim
            scores, rnn_out = self.model(trial_input, get_rnnout=True, h0=hn)
            DVs['rnn_out'][trial + 1] = rnn_out[1, 0, :].detach().cpu().numpy() # shape: hidden_dim
            DVs['scores'][trial + 1] = scores[1, 0, :].detach().cpu().numpy() # shape: output_dim
            DVs['choice_probs'][trial + 1] = scores[1, 0, :].softmax(dim=-1).detach().cpu().numpy() # shape: output_dim
            hn = rnn_out[-1:, :, :] # shape: seq_len=1, batch_size=1, hidden_dim
            update_session(session, n_trials, trial, pack_trial_event(c, obs, use_one_hot='false')) # cannot be one-hot when recording

        DVs['trial_log_likelihood'] = protected_log(DVs['choice_probs'][np.arange(n_trials), session['choices']])
        session_log_likelihood = np.sum(DVs['trial_log_likelihood'])

        if get_DVs:
            return DVs | {'session_log_likelihood': session_log_likelihood} | session
        else:
            return session_log_likelihood



def protected_log(x):
    """Return log of x protected against giving -inf for very small values of x."""
    return np.log(((1e-200) / 2) + (1 - (1e-200)) * x)

def choose(P):
    """Takes vector of probabilities P summing to 1, returns integer s with prob P[s]"""
    return sum(np.cumsum(P) < random())

def _tensor_structure_to_numpy(obj):
    """Convert all tensors (nested) in best_model_pass to numpy arrays."""
    if isinstance(obj, dict):
        obj_new = {k: _tensor_structure_to_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        obj_new = [_tensor_structure_to_numpy(v) for v in obj]
    elif isinstance(obj, torch.Tensor):
        obj_new = obj.detach().cpu().numpy()
        if obj_new.size == 1:  # transform 1-element array to scalar
            obj_new = obj_new.item()
    else:
        obj_new = obj
    return obj_new