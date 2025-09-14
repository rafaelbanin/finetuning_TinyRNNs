"""Add trainer related methods to RNNAgents.

Mostly the train method and the train_diagnose method. Agent data analysis should be minimal here.
"""
from .RNNAgent import RNNAgent, _tensor_structure_to_numpy
# from .BaseTrainer import BaseTrainer
import torch
torch.backends.cudnn.benchmark=True
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from copy import deepcopy
import os
import json
import joblib
import time
from path_settings import *
import numpy as np
from pathlib import Path
from tqdm import tqdm

class RNNAgentTrainer(RNNAgent):
    """
    A wrapper of RNNAgent, adding additional functions for training.

    Attributes:
        config: a dict of configuration
        optimizer: an optimizer for training
        penalized_weights: weights penalized by L1/L2 regularization (if any)
        penalized_weight_names: names of the penalized weights
        data: data can be directly bound before training (useful when multiprocessing)
    """

    def __init__(self, agent):
        """Initialize the trainer.

        Args:
            agent (RNNAgent): the agent to be wrapped.

        Should not call __init__ of the parent class, since it will initialize a new model.
        Notice here we cannot directly write self.save = agent.save,
        since the agent.save method just becomes a bound method, and the self in the agent.save method is the agent.
        """
        self.agent = agent
        attr_method_taken_from_agent = [
            'config', 'rnn_type', 'model', 'behav_loss_function', 'num_params', # attributes
            'load', 'forward', 'save_config' # agent methods; save is re-written
        ]
        for attr in attr_method_taken_from_agent:
            setattr(self, attr, getattr(agent, attr))

        config = self.config
        self.save_config()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        self.penalized_weight_names = []
        self.penalized_weights = []
        if 'l1_weight' in config and config['l1_weight']>0:
            assert 'penalized_weight' in config
            if config['penalized_weight'] == 'rec':
                # penalize recurrent weights, not input/output weights
                for name, x in self.model.named_parameters():
                    if 'weight' in name and 'rnn' in name:
                        self.penalized_weight_names.append(name)
                        self.penalized_weights.append(x)
            else:
                raise NotImplementedError


    # multiprocessing has complex interactions with __getattr__, so when uncommenting the following line be careful
    # https://stackoverflow.com/questions/62331047/why-am-i-receiving-a-recursion-error-with-multiprocessing
    # https://stackoverflow.com/questions/50156118/recursionerror-maximum-recursion-depth-exceeded-while-calling-a-python-object-w/50158865#50158865
    # def __getattr__(self, item):
    #     """If the attribute is not found in the trainer, search in the agent."""
    #     if item == 'agent':
    #         raise ValueError('Encountered infinite recursion. Check the call stack.')
    #     return getattr(self.agent, item)
    # def __getstate__(self):
    #     """Multiprocessing requires the object to be pickable.
    #     This function is called when pickling the object. Create this function to avoid call __getattr__ when pickling."""
    #     return self.__dict__
    # def __setstate__(self, state):
    #     """Multiprocessing requires the object to be pickable.
    #     This function is called when unpickling the object. Create this function to avoid call __getattr__ when unpickling."""
    #     self.__dict__.update(state)

    def bind_data(self, data):
        """Data is bound before call train method. Useful when multiprocessing."""
        self.data = data

    def train(self, data=None, verbose_level=1):
        """Train the model on some training data until overfit on the validation data. Early stop.

        Args:
            data (dict):
                data['train']: training data, containing input, target, mask
                data['val']: validation data, containing input, target, mask
                data['test']: test data, containing input, target, mask
                data['test'] can be the same as data['val'].
            verbose_level (optional):

        Returns:
            self: the trained agent
        """
        time_start = time.time()
        if data is None:
            data = self.data
            assert data is not None
        input_train, target_train, mask_train = data['train']['input'], data['train']['target'], data['train']['mask']
        input_val, target_val, mask_val = data['val']['input'], data['val']['target'], data['val']['mask']
        input_test, target_test, mask_test = data['test']['input'], data['test']['target'], data['test']['mask']
        best_loss = 1e+10
        best_model_pass = {}

        overfit_counter = 0
        best_state_dict = None

        train_loss_log = []
        val_loss_log = []
        test_loss_log = []
        separate_loss_log = {}
        def store_other_loss(separate_loss_log, data_pass, data_pass_name):
            # e.g., data_pass = val_pass, data_pass_name = 'val'
            for target_name in data_pass['all_target_names']:
                separate_loss_log.setdefault('behav_loss_' + target_name, {}).setdefault(data_pass_name, []).append(data_pass['behav_loss_' + target_name])

        if 'batch_size' in self.config:
            batch_size = self.config['batch_size']
        else:
            batch_size = 0
        if 'finetune' in self.config and self.config['finetune']:
            print('Finetune from the previous saved model...')
            if self.config['model_based'] == '100_pre_trained':
                mp = 'exp_finetuned_monkeyV/agent_name-Q(0)_seed0.rnn_type-GRU.hidden_dim-2.model_based-sintetic.trainval_percent-100/outerfold0_innerfold0_seed0'
            elif self.config['model_based'] ==  '70_pre_trained':
                mp = 'exp_finetuned_monkeyV/agent_name-Q(0)_seed0.rnn_type-GRU.hidden_dim-2.model_based-sintetic.trainval_percent-70/outerfold0_innerfold0_seed0'
            elif self.config['model_based'] ==  '50_pre_trained':
                mp = 'exp_finetuned_monkeyV/agent_name-Q(0)_seed0.rnn_type-GRU.hidden_dim-2.model_based-sintetic.trainval_percent-50/outerfold0_innerfold0_seed0'
            elif self.config['model_based'] ==  '20_pre_trained':
                mp = 'exp_finetuned_monkeyV/agent_name-Q(0)_seed0.rnn_type-GRU.hidden_dim-2.model_based-sintetic.trainval_percent-20/outerfold0_innerfold0_seed0'
            elif self.config['model_based'] ==  '10_pre_trained':
                mp = 'exp_finetuned_monkeyV/agent_name-Q(0)_seed0.rnn_type-GRU.hidden_dim-2.model_based-sintetic.trainval_percent-10/outerfold0_innerfold0_seed0'
            elif self.config['model_based'] ==  '30_pre_trained':
                mp = 'exp_finetuned_monkeyV/agent_name-Q(0)_seed0.rnn_type-GRU.hidden_dim-2.model_based-sintetic.trainval_percent-30/outerfold0_innerfold0_seed0'
            elif self.config['model_based'] ==  '40_pre_trained':
                mp = 'exp_finetuned_monkeyV/agent_name-Q(0)_seed0.rnn_type-GRU.hidden_dim-2.model_based-sintetic.trainval_percent-40/outerfold0_innerfold0_seed0'
            elif self.config['model_based'] ==  '60_pre_trained':
                mp = 'exp_finetuned_monkeyV/agent_name-Q(0)_seed0.rnn_type-GRU.hidden_dim-2.model_based-sintetic.trainval_percent-60/outerfold0_innerfold0_seed0'
            elif self.config['model_based'] ==  '80_pre_trained':
                mp = 'exp_finetuned_monkeyV/agent_name-Q(0)_seed0.rnn_type-GRU.hidden_dim-2.model_based-sintetic.trainval_percent-80/outerfold0_innerfold0_seed0'
            elif self.config['model_based'] ==  '90_pre_trained':
                mp = 'exp_finetuned_monkeyV/agent_name-Q(0)_seed0.rnn_type-GRU.hidden_dim-2.model_based-sintetic.trainval_percent-90/outerfold0_innerfold0_seed0'
            

            self.load(model_path=mp)
        self.use_amp = self.config.get('use_amp', False)
        if self.use_amp:
            print('Using AMP for training.')
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        with torch.autocast(device_type=self.config['device'], dtype=torch.float16, enabled=self.use_amp):
            for epoch in tqdm(range(self.config['max_epoch_num']+1)):
                # we can gather 1+max_epoch_num data points, since the first data point is the initial model before training
                current_epoch_pass = {}
                current_epoch_pass['epoch'] = epoch
                # losses for the model before this epoch's update, val/test/train is carefully aligned for the same model checkpoint
                current_epoch_pass['val'] = val_pass = self._eval_1step(input_val, target_val, mask_val)
                current_epoch_pass['test'] = test_pass = self._eval_1step(input_test, target_test, mask_test)
                # Imprime o resultado do teste inicial (epoch 0) de forma destacada
                if epoch == 0 and verbose_level >= 1 :
                    print(f"\n--- Resultado da Época 0 (Parâmetros Iniciais) ---")
                    print(f"    Val Loss: {val_pass['behav_loss']:.6f}")
                    print(f"    Test Loss: {test_pass['behav_loss']:.6f}")
                    print(f"---------------------------------------------------")
                val_loss_log.append(val_pass['behav_loss'])
                store_other_loss(separate_loss_log, val_pass, 'val')
                test_loss_log.append(test_pass['behav_loss'])
                store_other_loss(separate_loss_log, test_pass, 'test')
                if val_pass['behav_loss'] < best_loss:
                    # the model before update is better than the previous best model
                    best_loss = val_pass['behav_loss']
                    best_model_pass = current_epoch_pass # the training loss of the model before update will be added later
                    best_state_dict = deepcopy(self.model.state_dict())
                    overfit_counter = 0
                else:
                    overfit_counter += 1
                # the training loss is computed from the model just before the gradient update
                # after _train_1step, the model is updated
                if batch_size == 0: #  the whole training dataset as a batch
                    current_epoch_pass['train'] = train_pass = self._train_1step(input_train, target_train, mask_train)
                else:
                    assert not isinstance(target_train, tuple) # not support tuple target for now
                    current_epoch_pass['train'] = train_pass = self._eval_1step(input_train, target_train, mask_train)
                    # train for 1 epoch with a random batch order
                    batch_num = int(np.ceil(input_train.shape[1] / batch_size))
                    batch_order = np.random.permutation(batch_num)
                    for batch_idx in batch_order:
                        batch_start = int(batch_idx * batch_size)
                        batch_end = min(int((batch_idx + 1) * batch_size), input_train.shape[1])
                        batch_input = input_train[:, batch_start:batch_end]
                        batch_target = target_train[:, batch_start:batch_end]
                        if isinstance(mask_train, tuple):
                            batch_mask = (mask_train[0][:, batch_start:batch_end], mask_train[1][:, batch_start:batch_end])
                        else:
                            batch_mask = mask_train[:, batch_start:batch_end]
                        self._train_1step(batch_input, batch_target, batch_mask)

                train_loss_log.append(train_pass['behav_loss'])
                store_other_loss(separate_loss_log, train_pass, 'train')
                if verbose_level >=3 or epoch % (self.config['max_epoch_num'] // 100) == 0:
                    print('Epoch %d, train loss %.4f, val loss %.4f, test loss %.4f' % (epoch, train_pass['behav_loss'], val_pass['behav_loss'], test_pass['behav_loss']))

                if overfit_counter > self.config['early_stop_counter']:
                    break

        assert len(train_loss_log) == len(val_loss_log) == len(test_loss_log)
        assert 'train' in best_model_pass, val_pass['behav_loss']
        best_model_pass['train']['loss_log'] = train_loss_log
        best_model_pass['val']['loss_log'] = val_loss_log
        best_model_pass['test']['loss_log'] = test_loss_log
        if len(separate_loss_log) > 1:
            best_model_pass['separate_loss_log'] = separate_loss_log
        best_model_pass = _tensor_structure_to_numpy(best_model_pass)
        self.best_model_pass = best_model_pass
        self.save(params=best_state_dict, verbose=False)
        self.load(self.config['model_path']) # return the best one for using afterwards (not the last one)
        if verbose_level>0:
            print('Model',Path(self.config['model_path']).name,
                  'Training done. time cost:',time.time() - time_start,
                  'best train loss:', best_model_pass['train']['behav_loss'],
                  'best val loss:', best_model_pass['val']['behav_loss'],
                  'best test loss:', best_model_pass['test']['behav_loss'])
            if len(best_model_pass['train']['all_target_names']) > 1:
                for target_name in best_model_pass['train']['all_target_names']:
                    for data_pass_name in ['train', 'val', 'test']:
                        if 'behav_loss_'+target_name in best_model_pass[data_pass_name]:
                            print(data_pass_name, 'loss for', target_name, ':', best_model_pass[data_pass_name]['behav_loss_'+target_name], end=' ')
                    print()
        return self

    def save(self, params=None, verbose=False):
        """Save the model to the disk.

        Args:
            params (optional): the parameters to save. If None, save the self's parameters.
            verbose (optional): print the path of the saved model.
        """
        self.agent.save(params=params, verbose=verbose)
        if hasattr(self, 'best_model_pass'):
            if self.config['save_model_pass'] == 'full':
                save_model_pass = self.best_model_pass
            elif self.config['save_model_pass'] == 'minimal':  # only save the likelihoods to save space
                save_model_pass = {
                    'train': {
                        'behav_loss': self.best_model_pass['train']['behav_loss'],
                        'loss_log': self.best_model_pass['train']['loss_log']},
                    'val': {
                        'behav_loss': self.best_model_pass['val']['behav_loss'],
                        'loss_log': self.best_model_pass['val']['loss_log']},
                    'test': {
                        'behav_loss': self.best_model_pass['test']['behav_loss'],
                        'loss_log': self.best_model_pass['test']['loss_log']}
                }
            elif self.config['save_model_pass'] == 'none':
                return
            else:
                raise ValueError('Unknown save_model_pass:', self.config['save_model_pass'])
            joblib.dump(save_model_pass, MODEL_SAVE_PATH / self.config['model_path'] / 'best_pass.pkl')


    def training_diagnose(self):
        """Diagnosis of training results, e.g. plot all losses over time.
        """
        def wrap_plot_loss(train_loss_log, val_loss_log, test_loss_log=None, fname='loss_log'):
            epoch_num = len(train_loss_log)
            plt.figure()
            plt.plot(range(epoch_num), train_loss_log, label='train')
            plt.plot(range(epoch_num), val_loss_log, label='val')
            plt.vlines(self.best_model_pass['epoch'], np.min(val_loss_log), np.max(val_loss_log), colors='k',
                       label='earlystop')
            if test_loss_log is not None:
                plt.plot(range(epoch_num), test_loss_log, label='test')
            plt.legend()
            plt.xlabel('epoch')
            plt.ylabel('CE loss')
            plt.savefig(MODEL_SAVE_PATH / self.config['model_path'] / f'{fname}.pdf')
            plt.close()

        if 'training_diagnose' in self.config and self.config['training_diagnose'] is not None and len(self.config['training_diagnose']) > 0:
            if not hasattr(self, 'best_model_pass'):
                return
            import matplotlib.pyplot as plt
            if 'plot_loss' in self.config['training_diagnose']:
                train_loss_log = self.best_model_pass['train']['loss_log']
                val_loss_log = self.best_model_pass['val']['loss_log']
                test_loss_log = self.best_model_pass['test']['loss_log'] if 'test' in self.best_model_pass else None
                wrap_plot_loss(train_loss_log, val_loss_log, test_loss_log, fname='loss_log')
                if 'separate_loss_log' in self.best_model_pass and len(self.best_model_pass['train']['all_target_names'])>1:
                    for target_name in self.best_model_pass['train']['all_target_names']:
                        train_loss_log = self.best_model_pass['separate_loss_log']['behav_loss_' + target_name]['train']
                        val_loss_log = self.best_model_pass['separate_loss_log']['behav_loss_' + target_name]['val']
                        test_loss_log = self.best_model_pass['separate_loss_log']['behav_loss_' + target_name]['test'] if 'test' in self.best_model_pass['separate_loss_log']['behav_loss_' + target_name] else None
                        wrap_plot_loss(train_loss_log, val_loss_log, test_loss_log, fname='loss_log_'+target_name)

    def _train_1step(self, input, target, mask, h0=None):
        """One step of optimization on training dataset."""
        self.model.zero_grad()
        model_pass = self._compare_to_target(input, target, mask, h0=h0)
        l1_loss = torch.tensor(0., requires_grad=True)
        if len(self.penalized_weights) > 0:
            for x in self.penalized_weights:
                l1_loss = l1_loss + self.config['l1_weight'] * torch.linalg.vector_norm(x, ord=1)
        model_pass['l1_loss'] = l1_loss
        total_loss = model_pass['behav_loss'] + l1_loss
        # total_loss.backward()
        self.scaler.scale(total_loss).backward()  # loss.backward()
        # Unscales the gradients of optimizer's assigned parameters in-place
        self.scaler.unscale_(self.optimizer)
        # Since the gradients of optimizer's assigned parameters are now unscaled, clips as usual.
        if 'grad_clip' in self.config:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
        # Update weights
        self.scaler.step(self.optimizer)  # optimizer.step()
        # Updates the scale for next iteration.
        self.scaler.update()
        # self.optimizer.step()
        return model_pass
