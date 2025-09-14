"""
Customized RNN networks.
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import random
import os
import gc
import platform
# FORCE_USING_TORCH_SCRIPT = False
#
# if FORCE_USING_TORCH_SCRIPT is True or (FORCE_USING_TORCH_SCRIPT is None and platform.system() == 'Windows'):
#     from .custom_lstms import RNNLayer_custom
#     # torchscript can save ~25% time for the current layers
#     # cannot work together with multiprocessing, can be used for single process training
#     # current code will trigger an internal bug in pytorch on Linux
# else:
from .custom_rnn_layers import RNNLayer_custom
import torch.jit as jit

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MaskedBlockLinear(nn.Module):
    def __init__(self, hidden_dim, output_dim, block_num):
        super(MaskedBlockLinear, self).__init__()
        assert hidden_dim <= output_dim, (hidden_dim, output_dim)
        assert hidden_dim % block_num == 0, "hidden_dim should be divisible by block_num"
        assert output_dim % hidden_dim == 0, "output_dim should be a multiple of hidden_dim"

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.block_num = block_num

        self.weight = nn.Parameter(torch.Tensor(output_dim, hidden_dim))
        self.bias = nn.Parameter(torch.Tensor(output_dim))

        self.mask = self.generate_block_mask(hidden_dim, output_dim, block_num)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)
        self.weight.data *= self.mask

    def generate_block_mask(self, hidden_dim, output_dim, block_num):
        mask = torch.zeros(output_dim, hidden_dim, device=self.weight.device)
        hidden_block_size = hidden_dim // block_num
        output_block_size = output_dim // block_num
        for i in range(block_num):
            mask[i * output_block_size:(i + 1) * output_block_size,
            i * hidden_block_size:(i + 1) * hidden_block_size] = 1
        return mask

    def forward(self, x):
        if self.mask.device != self.weight.device:
            self.mask = self.mask.to(self.weight.device)
        masked_weight = self.weight * self.mask
        return F.linear(x, masked_weight, self.bias)

class RNNnet(nn.Module):
# class RNNnet(jit.ScriptModule):
    """ A RNN network: input layer + recurrent layer + a readout layer. Supports multiple embedding layers.

    Attributes:
        input_dim:
        hidden_dim:
        output_dim:
        output_h0: whether the output of the network should contain the one from initial networks' hidden state
        rnn: the recurrent layer
        h0: the initial networks' hidden state
        readout_FC: whether the readout layer is full connected or not
        lin: the full connected readout layer
        lin_coef: the inverse temperature of a direct readout layer
    """
    def __init__(self, input_dim, hidden_dim, output_dim, readout_FC=True, trainable_h0=False, rnn_type='GRU', output_h0='False',**kwargs):
        """
        Args:
            input_dim:
            hidden_dim:
            output_dim:
            output_h0:
            readout_FC:
            rnn_type:
            trainable_h0: the agent's initial hidden state trainable or not
        """
        super(RNNnet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_h0 = output_h0
        self.include_embedding = False
        # self.num_embedding_channels = 0
        # self.embedding_dim = [0]
        # self.embedding = [nn.Embedding(0, 0)]
        if 'include_embedding' in kwargs and kwargs['include_embedding']:
            self.include_embedding = True
            if 'num_embedding_channels' in kwargs and kwargs['num_embedding_channels']:
                # multiple embedding layers
                self.num_embedding_channels = kwargs['num_embedding_channels']
                self.num_embeddings = []
                self.embedding_dim = []
                self.embedding = []

                for channel in range(self.num_embedding_channels):
                    self.num_embeddings.append(kwargs['num_embeddings_c'+str(channel)])
                    self.embedding_dim.append(kwargs['embedding_dim_c'+str(channel)])
                    embedding_init = kwargs['embedding_init_c'+str(channel)]
                    if embedding_init == 'zero':
                        zero_embedding = torch.zeros(self.num_embeddings[channel], self.embedding_dim[channel])
                        self.embedding.append(nn.Embedding(self.num_embeddings[channel], self.embedding_dim[channel], _weight=zero_embedding))
                    elif embedding_init == 'id':
                        assert self.num_embeddings[channel] == self.embedding_dim[channel], (self.num_embeddings[channel], self.embedding_dim[channel])
                        id_embedding = torch.eye(self.num_embeddings[channel], self.embedding_dim[channel])
                        self.embedding.append(nn.Embedding(self.num_embeddings[channel], self.embedding_dim[channel], _weight=id_embedding))
                    else:
                        raise NotImplementedError
                    self.__setattr__('embedding_c'+str(channel), self.embedding[channel]) # register the embedding layer to the model
            else:
                # single embedding layer
                self.num_embedding_channels = 1
                self.num_embeddings = [kwargs['num_embeddings']] # the number of subjects
                self.embedding_dim = [kwargs['embedding_dim']] # the dimension of the embedding
                zero_embedding = torch.zeros(self.num_embeddings[0], self.embedding_dim[0])
                self.embedding = nn.Embedding(self.num_embeddings[0], self.embedding_dim[0], _weight=zero_embedding)
            input_dim += np.sum(self.embedding_dim)
        # self.input_dim is not changed, but the input_dim in this function is changed

        if rnn_type == 'GRU': # official GRU implementation
            self.rnn = nn.GRU(input_dim, hidden_dim)
        else: # customized RNN layers
            self.rnn = RNNLayer_custom(input_dim, hidden_dim, rnn_type=rnn_type, **kwargs)
        self.readout_FC = readout_FC

        if readout_FC:
            self.lin = nn.Linear(hidden_dim, output_dim)
            assert 'readout_block_num' not in kwargs, 'readout_block_num is not supported when readout_FC=True'
        else:
            if 'readout_block_num' in kwargs:
                self.readout_block_num = kwargs['readout_block_num']
                self.lin = MaskedBlockLinear(hidden_dim, output_dim, self.readout_block_num)
            else:
                assert hidden_dim == output_dim, (hidden_dim, output_dim)
                self.lin_coef = nn.Parameter(torch.ones(1,1,1))

        if trainable_h0:
            self.h0 = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        else:
            self.h0 = torch.zeros(1, 1, hidden_dim).double()
        self.dummy_param = nn.Parameter(torch.empty(0)) # a dummy parameter to store the device of the model

        if 'output_layer_num' in kwargs and kwargs['output_layer_num']>1:
            self.output_layer_num = kwargs['output_layer_num']
            assert self.output_layer_num == 2, 'only support 2 output layers for now'
            self.output_dim1 = kwargs['output_dim1']
            self.lin1 = nn.Linear(hidden_dim, self.output_dim1)
        else:
            self.output_layer_num = 1

    # @jit.script_method
    def forward(self, input, get_rnnout=False, h0=None):
        """
        Args:
            input: shape: seq_len, batch_size, input_size
            get_rnnout: whether the internal states of rnn should be outputted
            h0: whether use a customized h0 or default h0
                shape: seq_len=1, batch_size=1, hidden_dim

        Returns:
            Return the final output of the RNN
            Also return the internal states if get_rnnout=True
        """
        model_device = self.dummy_param.device
        if self.h0.device != model_device: # move h0 to the same device as the model if h0 is not Parameter
            self.h0 = self.h0.to(model_device) # this should be done before the first forward pass
        if h0 is None:
            h0 = self.h0
        assert input.device == model_device, (input.device, model_device)
        assert h0.device == model_device, (h0.device, model_device)
        seq_len, batch_size, input_dim = input.shape
        h0_expand = h0.repeat(1, batch_size, 1) # h0 is the same for each sample in the batch
        if self.include_embedding:
            nbc = self.num_embedding_channels
            input, embedding_inputs = input[..., :-nbc], input[..., -nbc:]
            assert input.shape[-1] == self.input_dim, (input.shape, self.embedding_dim)
            final_input_list = [input]
            for channel in range(nbc):
                embedding_input = embedding_inputs[...,channel].long()

                embedding = self.embedding[channel](embedding_input) if isinstance(self.embedding, list) else self.embedding(embedding_input)
                assert embedding.shape == (seq_len, batch_size, self.embedding_dim[channel]), (embedding.shape, (seq_len, batch_size, self.embedding_dim[channel]))
                final_input_list.append(embedding)
            input = torch.cat(final_input_list, -1)
        rnn_out, hn = self.rnn(input, h0_expand)  # rnn_out shape: seq_len, batch, hidden_size
        if self.output_h0:
            rnn_out = torch.cat((h0_expand, rnn_out), 0)
            seq_len += 1
        if self.readout_FC or hasattr(self, 'readout_block_num'):
            scores = self.lin(rnn_out.view(seq_len * batch_size, self.hidden_dim))
        else:
            scores = self.lin_coef * rnn_out
        scores = scores.view(seq_len, batch_size, self.output_dim)

        if self.output_layer_num > 1:
            scores1 = self.lin1(rnn_out.view(seq_len * batch_size, self.hidden_dim)).view(seq_len, batch_size, self.output_dim1)
            scores = [scores, scores1]
        if get_rnnout:
            return scores, rnn_out
        return scores
