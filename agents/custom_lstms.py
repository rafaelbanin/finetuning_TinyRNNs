""" Custom RNN implementation for PyTorch using TorchScript (do not work well on clusters)."""
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.jit as jit
import warnings
from collections import namedtuple
from typing import List, Tuple
from torch import Tensor
import numbers
import math
'''
Some helper classes for writing custom TorchScript LSTMs.

Goals:
- Classes are easy to read, use, and extend
- Performance of custom LSTMs approach fused-kernel-levels of speed.

A few notes about features we could add to clean up the below code:
- Support enumerate with nn.ModuleList:
  https://github.com/pytorch/pytorch/issues/14471
- Support enumerate/zip with lists:
  https://github.com/pytorch/pytorch/issues/15952
- Support overriding of class methods:
  https://github.com/pytorch/pytorch/issues/10733
- Support passing around user-defined namedtuple types for readability
- Support slicing w/ range. It enables reversing lists easily.
  https://github.com/pytorch/pytorch/issues/10774
- Multiline type annotations. List[List[Tuple[Tensor,Tensor]]] is verbose
  https://github.com/pytorch/pytorch/pull/14922
'''


def script_lstm(input_size, hidden_size, num_layers, bias=True,
                batch_first=False, dropout=False, bidirectional=False):
    """Returns a ScriptModule that mimics a PyTorch native LSTM."""

    # The following are not implemented.
    assert bias
    assert not batch_first

    if bidirectional:
        stack_type = StackedLSTM2
        layer_type = BidirLSTMLayer
        dirs = 2
    elif dropout:
        stack_type = StackedLSTMWithDropout
        layer_type = LSTMLayer
        dirs = 1
    else:
        stack_type = StackedLSTM
        layer_type = LSTMLayer
        dirs = 1

    return stack_type(num_layers, layer_type,
                      first_layer_args=[LSTMCell, input_size, hidden_size],
                      other_layer_args=[LSTMCell, hidden_size * dirs,
                                        hidden_size])


def script_lnlstm(input_size, hidden_size, num_layers, bias=True,
                  batch_first=False, dropout=False, bidirectional=False,
                  decompose_layernorm=False):
    '''Returns a ScriptModule that mimics a PyTorch native LSTM.'''

    # The following are not implemented.
    assert bias
    assert not batch_first
    assert not dropout

    if bidirectional:
        stack_type = StackedLSTM2
        layer_type = BidirLSTMLayer
        dirs = 2
    else:
        stack_type = StackedLSTM
        layer_type = LSTMLayer
        dirs = 1

    return stack_type(num_layers, layer_type,
                      first_layer_args=[LayerNormLSTMCell, input_size, hidden_size,
                                        decompose_layernorm],
                      other_layer_args=[LayerNormLSTMCell, hidden_size * dirs,
                                        hidden_size, decompose_layernorm])


LSTMState = namedtuple('LSTMState', ['hx', 'cx'])


def reverse(lst: List[Tensor]) -> List[Tensor]:
    return lst[::-1]


class LSTMCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = Parameter(torch.randn(4 * hidden_size))

    @jit.script_method
    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = state
        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)


class GRUCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(3 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(3 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(3 * hidden_size))
        self.bias_hh = Parameter(torch.randn(3 * hidden_size))

    @jit.script_method
    def forward(self, input: Tensor, state: Tuple[Tensor]) -> Tuple[Tensor]:
        hx = state[0]
        gates_i = torch.mm(input, self.weight_ih.t()) + self.bias_ih
        gates_h = torch.mm(hx, self.weight_hh.t()) + self.bias_hh
        reset_i, update_i, new_i = gates_i.chunk(3, 1)
        reset_h, update_h, new_h = gates_h.chunk(3, 1)
        reset_gate = torch.sigmoid(reset_i + reset_h)
        update_gate = torch.sigmoid(update_i + update_h)
        new_gate = torch.tanh(new_i + reset_gate * new_h)
        hy = update_gate * hx + (1 - update_gate) * new_gate

        return hy,

class MIGRUCell(jit.ScriptModule):
    """multiplicative integration GRU"""
    def __init__(self, input_size, hidden_size):
        super(MIGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        r2 = math.sqrt(1/hidden_size)
        r1 = - r2
        self.weight_ih = Parameter(self.uniform_parameter(r1, r2, (3 * hidden_size, input_size)))
        self.weight_hh = Parameter(self.uniform_parameter(r1, r2, (3 * hidden_size, hidden_size)))
        self.bias_ih = Parameter(self.uniform_parameter(r1, r2, (3 * hidden_size,))/100) # close to 0
        self.bias_hh = Parameter(self.uniform_parameter(r1, r2, (3 * hidden_size,))/100)
        self.alpha_ih = Parameter(torch.zeros(3 * hidden_size)) # randn
        self.beta_i = Parameter(torch.ones(3 * hidden_size)) # randn
        self.beta_h = Parameter(torch.ones(3 * hidden_size)) # randn

    def uniform_parameter(self, r1, r2, size):
        temp = (r2 - r1) * torch.rand(*size) + r1
        return temp

    @jit.script_method
    def forward(self, input: Tensor, state: Tuple[Tensor]) -> Tuple[Tensor]:
        hx = state[0]
        input_cur = torch.mm(input, self.weight_ih.t())
        hx_cur = torch.mm(hx, self.weight_hh.t())
        gates_i = self.beta_i * input_cur + self.bias_ih
        gates_h = self.beta_h * hx_cur + self.bias_hh
        gates_ih = self.alpha_ih * input_cur * hx_cur

        reset_i, update_i, new_i = gates_i.chunk(3, 1)
        reset_h, update_h, new_h = gates_h.chunk(3, 1)
        reset_ih, update_ih, new_ih = gates_ih.chunk(3, 1)
        reset_gate = torch.sigmoid(reset_ih + reset_i + reset_h)
        update_gate = torch.sigmoid(update_ih + update_i + update_h)
        new_gate = torch.tanh(new_ih * reset_gate + new_i + reset_gate * new_h)
        hy = update_gate * hx + (1 - update_gate) * new_gate

        return hy,

class SGRUCell(jit.ScriptModule):
    """switching GRU.

    Modification of the original GRU. Because the input is one-hot (for combinations of discrete input features),
    will use different recurrent weight depending on the input.
    """
    def __init__(self, input_size, hidden_size):
        super(SGRUCell, self).__init__()
        self.input_size = input_size # input_size is for one-hot encoding
        self.hidden_size = hidden_size
        #self.weight_ih = Parameter(torch.randn(3 * hidden_size, input_size)) # TODO: use this for continous input dimension
        self.weight_hh = Parameter(torch.randn(3 * hidden_size, hidden_size, input_size)) # 3H*H*I
        self.bias_ih = Parameter(torch.randn(3 * hidden_size, input_size)) # 3H*I
        self.bias_hh = Parameter(torch.randn(3 * hidden_size, input_size)) # 3H*I

    @jit.script_method
    def forward(self, input: Tensor, state: Tuple[Tensor]) -> Tuple[Tensor]:
        # input is one-hot: batch_size, input_size
        assert (input.sum(-1) == 1).all()
        hx = state[0] # B*H
        trial_weight_hh = (self.weight_hh[None, :,:,:] * input[:, None, None, :]).sum(-1) # 1*3H*H*I, B*1*1*I-> B*3H*H
        rec_temp = (trial_weight_hh * hx[:, None, :]).sum(-1) # B*3H*H, B*1*H->B*3H
        trial_bias_ih = (self.bias_ih[None, :,:] * input[:, None, :]).sum(-1) # 1*3H*I, B*1*I-> B*3H
        trial_bias_hh = (self.bias_hh[None, :,:] * input[:, None, :]).sum(-1) # 1*3H*I, B*1*I-> B*3H
        gates_i = trial_bias_ih
        gates_h = rec_temp + trial_bias_hh
        reset_i, update_i, new_i = gates_i.chunk(3, 1)
        reset_h, update_h, new_h = gates_h.chunk(3, 1)
        reset_gate = torch.sigmoid(reset_i + reset_h)
        update_gate = torch.sigmoid(update_i + update_h)
        new_gate = torch.tanh(new_i + reset_gate * new_h)
        hy = update_gate * hx + (1 - update_gate) * new_gate

        return hy,


class PNRCell(jit.ScriptModule):
    """Polynomial regression model.
    Each time step is a polynomial transfomation.
    Only support hidden_size<=2 and order<=3 for now.

    Attributes:
        po: polynomial_order
    """
    def __init__(self, input_size, hidden_size, polynomial_order=0):
        super(PNRCell, self).__init__()
        self.input_size = input_size # input_size is for one-hot
        self.hidden_size = hidden_size
        self.po = polynomial_order
        assert polynomial_order>0, "polynomial_order not provided"
        feature_size = 0
        if self.hidden_size == 1:
            feature_size += self.po
        elif self.hidden_size == 2:
            if self.po >= 1:
                feature_size += 2
            if self.po >= 2:
                feature_size += 3
            if self.po >= 3:
                feature_size += 4
            if self.po >= 4:
                raise NotImplementedError
        else:
            raise NotImplementedError
        self.feature_size = feature_size
        self.weight = Parameter(torch.zeros(hidden_size, feature_size, input_size)) # H*F*I
        self.bias = Parameter(torch.zeros(hidden_size, input_size)) # H*I

    @jit.script_method
    def forward(self, input: Tensor, state: Tuple[Tensor]) -> Tuple[Tensor]:
        # input is one-hot: batch_size, input_size
        assert (input.sum(-1) == 1).all()
        trial_weight = (self.weight[None, :,:,:] * input[:, None, None, :]).sum(-1) # 1*H*F*I, B*1*1*I-> B*H*F
        trial_bias = (self.bias[None, :,:] * input[:, None, :]).sum(-1) # 1*H*I, B*1*I-> B*H
        hx = state[0] # B*H
        batch_size = hx.shape[0]
        features = []
        if self.hidden_size == 1:
            h1 = hx[:, 0]
            if self.po >= 1:
                features += [h1]
            if self.po >= 2:
                features += [h1**2]
            if self.po >= 3:
                features += [h1**3]
            if self.po >= 4:
                raise NotImplementedError
        elif self.hidden_size == 2:
            h1 = hx[:, 0]
            h2 = hx[:, 1]
            if self.po >= 1:
                features += [h1, h2]
            if self.po >= 2:
                features += [h1**2, h2**2, h1*h2]
            if self.po >= 3:
                features += [h1**3, h1**2*h2, h1*h2**2,h2**3]
            if self.po >= 4:
                raise NotImplementedError
        elif self.hidden_size == 3:
            h1 = hx[:, 0]
            h2 = hx[:, 1]
            h3 = hx[:, 2]
            if self.po >= 1:
                features += [h1, h2, h3]
            if self.po >= 2:
                features += [h1**2, h2**2, h3**2, h1*h2, h2*h3, h1*h3]
            if self.po >= 3:
                raise NotImplementedError
        else:
            raise NotImplementedError
        features = torch.stack(features, dim=1) # B, F
        rec_temp = (trial_weight * features[:, None, :]).sum(-1) # B*H*F, B*1*F->B*H
        hy = hx + rec_temp + trial_bias

        return hy,

class LayerNorm(jit.ScriptModule):
    def __init__(self, normalized_shape):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        # XXX: This is true for our LSTM / NLP use case and helps simplify code
        assert len(normalized_shape) == 1

        self.weight = Parameter(torch.ones(normalized_shape))
        self.bias = Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    @jit.script_method
    def compute_layernorm_stats(self, input):
        mu = input.mean(-1, keepdim=True)
        sigma = input.std(-1, keepdim=True, unbiased=False)
        return mu, sigma

    @jit.script_method
    def forward(self, input):
        mu, sigma = self.compute_layernorm_stats(input)
        return (input - mu) / sigma * self.weight + self.bias


class LayerNormLSTMCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size, decompose_layernorm=False):
        super(LayerNormLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        # The layernorms provide learnable biases

        if decompose_layernorm:
            ln = LayerNorm
        else:
            ln = nn.LayerNorm

        self.layernorm_i = ln(4 * hidden_size)
        self.layernorm_h = ln(4 * hidden_size)
        self.layernorm_c = ln(hidden_size)

    @jit.script_method
    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = state
        igates = self.layernorm_i(torch.mm(input, self.weight_ih.t()))
        hgates = self.layernorm_h(torch.mm(hx, self.weight_hh.t()))
        gates = igates + hgates
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = self.layernorm_c((forgetgate * cx) + (ingate * cellgate))
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)


class LSTMLayer(jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super(LSTMLayer, self).__init__()
        self.cell = cell(*cell_args)

    @jit.script_method
    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        inputs = input.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs), state


class RNNLayer_(jit.ScriptModule):
    """Customized RNN layer in TorchScript format.

    Attributes:
        rnn_type:
        rnncell: a cell responsible for a single time step.
    """
    def __init__(self, *cell_args, **kwargs):
        super(RNNLayer_, self).__init__()
        self.rnn_type = kwargs['rnn_type']
        if self.rnn_type == 'SGRU':
            self.rnncell = SGRUCell(*cell_args)
        elif self.rnn_type == 'MIGRU':
            self.rnncell = MIGRUCell(*cell_args)
        elif self.rnn_type == 'GRU':
            self.rnncell = GRUCell(*cell_args)
        elif 'PNR' in self.rnn_type:
            self.rnncell = PNRCell(*cell_args, polynomial_order=kwargs['polynomial_order'])
        else:
            print(self.rnn_type)
            raise NotImplementedError

    @jit.script_method
    def forward(self, input: Tensor, state: Tuple[Tensor]) -> Tuple[Tensor, Tuple[Tensor]]:
        """Run a RNN cell for several time steps.

        Args:
            input: shape: seq_len, batch_size, input_dim
            state: single-element tuple, shape: batch_size, hidden_dim

        Returns:
            outputs: shape: seq_len, batch_size, output_dim
            state: final state after seeing all inputs
        """
        inputs = input.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            state = self.rnncell(inputs[i], state)
            out = state[0]
            outputs += [out]
        return torch.stack(outputs), state

class RNNLayer_custom(nn.Module):
    """A wrapper making sure the input & output format of these TorchScript rnn layers consistent with official implementation.
    """
    def __init__(self, *cell_args, **kwargs):
        super(RNNLayer_custom, self).__init__()
        self.rnn = RNNLayer_(*cell_args, **kwargs)

    def forward(self, input: Tensor, state: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Take the standard torch format input and transform it into TorchScript format.
        Run the TorchScript style layers.
        Transform the output back to the standard format.
        """
        if len(state.shape) == 3: # when the state shape is seq_len, batch_size, hidden_dim
            assert state.shape[0] == 1
            state = state[0] # remove seq_len dim (=1)
            dim_flag = True
        else: # when the state shape is batch_size, hidden_dim
            dim_flag = False
        outputs, state = self.rnn(input, (state,)) # making state a one-element tuple
        state = state[0] # extract state from tuple
        if dim_flag: # make sure the output state has the same format as the previous time point
            state = state[None, ...] # insert seq_len=1 dim back
        return outputs, state


class ReverseLSTMLayer(jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super(ReverseLSTMLayer, self).__init__()
        self.cell = cell(*cell_args)

    @jit.script_method
    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        inputs = reverse(input.unbind(0))
        outputs = jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(reverse(outputs)), state


class BidirLSTMLayer(jit.ScriptModule):
    __constants__ = ['directions']

    def __init__(self, cell, *cell_args):
        super(BidirLSTMLayer, self).__init__()
        self.directions = nn.ModuleList([
            LSTMLayer(cell, *cell_args),
            ReverseLSTMLayer(cell, *cell_args),
        ])

    @jit.script_method
    def forward(self, input: Tensor, states: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        # List[LSTMState]: [forward LSTMState, backward LSTMState]
        outputs = jit.annotate(List[Tensor], [])
        output_states = jit.annotate(List[Tuple[Tensor, Tensor]], [])
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        for direction in self.directions:
            state = states[i]
            out, out_state = direction(input, state)
            outputs += [out]
            output_states += [out_state]
            i += 1
        return torch.cat(outputs, -1), output_states


def init_stacked_lstm(num_layers, layer, first_layer_args, other_layer_args):
    layers = [layer(*first_layer_args)] + [layer(*other_layer_args)
                                           for _ in range(num_layers - 1)]
    return nn.ModuleList(layers)


class StackedLSTM(jit.ScriptModule):
    __constants__ = ['layers']  # Necessary for iterating through self.layers

    def __init__(self, num_layers, layer, first_layer_args, other_layer_args):
        super(StackedLSTM, self).__init__()
        self.layers = init_stacked_lstm(num_layers, layer, first_layer_args,
                                        other_layer_args)

    @jit.script_method
    def forward(self, input: Tensor, states: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        # List[LSTMState]: One state per layer
        output_states = jit.annotate(List[Tuple[Tensor, Tensor]], [])
        output = input
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        for rnn_layer in self.layers:
            state = states[i]
            output, out_state = rnn_layer(output, state)
            output_states += [out_state]
            i += 1
        return output, output_states


# Differs from StackedLSTM in that its forward method takes
# List[List[Tuple[Tensor,Tensor]]]. It would be nice to subclass StackedLSTM
# except we don't support overriding script methods.
# https://github.com/pytorch/pytorch/issues/10733
class StackedLSTM2(jit.ScriptModule):
    __constants__ = ['layers']  # Necessary for iterating through self.layers

    def __init__(self, num_layers, layer, first_layer_args, other_layer_args):
        super(StackedLSTM2, self).__init__()
        self.layers = init_stacked_lstm(num_layers, layer, first_layer_args,
                                        other_layer_args)

    @jit.script_method
    def forward(self, input: Tensor, states: List[List[Tuple[Tensor, Tensor]]]) -> Tuple[Tensor, List[List[Tuple[Tensor, Tensor]]]]:
        # List[List[LSTMState]]: The outer list is for layers,
        #                        inner list is for directions.
        output_states = jit.annotate(List[List[Tuple[Tensor, Tensor]]], [])
        output = input
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        for rnn_layer in self.layers:
            state = states[i]
            output, out_state = rnn_layer(output, state)
            output_states += [out_state]
            i += 1
        return output, output_states


class StackedLSTMWithDropout(jit.ScriptModule):
    # Necessary for iterating through self.layers and dropout support
    __constants__ = ['layers', 'num_layers']

    def __init__(self, num_layers, layer, first_layer_args, other_layer_args):
        super(StackedLSTMWithDropout, self).__init__()
        self.layers = init_stacked_lstm(num_layers, layer, first_layer_args,
                                        other_layer_args)
        # Introduces a Dropout layer on the outputs of each LSTM layer except
        # the last layer, with dropout probability = 0.4.
        self.num_layers = num_layers

        if (num_layers == 1):
            warnings.warn("dropout lstm adds dropout layers after all but last "
                          "recurrent layer, it expects num_layers greater than "
                          "1, but got num_layers = 1")

        self.dropout_layer = nn.Dropout(0.4)

    @jit.script_method
    def forward(self, input: Tensor, states: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        # List[LSTMState]: One state per layer
        output_states = jit.annotate(List[Tuple[Tensor, Tensor]], [])
        output = input
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        for rnn_layer in self.layers:
            state = states[i]
            output, out_state = rnn_layer(output, state)
            # Apply the dropout layer except the last layer
            if i < self.num_layers - 1:
                output = self.dropout_layer(output)
            output_states += [out_state]
            i += 1
        return output, output_states


def flatten_states(states):
    states = list(zip(*states))
    assert len(states) == 2
    return [torch.stack(state) for state in states]


def double_flatten_states(states):
    # XXX: Can probably write this in a nicer way
    states = flatten_states([flatten_states(inner) for inner in states])
    return [hidden.view([-1] + list(hidden.shape[2:])) for hidden in states]


def test_script_rnn_layer(seq_len, batch, input_size, hidden_size):
    inp = torch.randn(seq_len, batch, input_size)
    state = LSTMState(torch.randn(batch, hidden_size),
                      torch.randn(batch, hidden_size))
    rnn = LSTMLayer(LSTMCell, input_size, hidden_size)
    out, out_state = rnn(inp, state)

    # Control: pytorch native LSTM
    lstm = nn.LSTM(input_size, hidden_size, 1)
    lstm_state = LSTMState(state.hx.unsqueeze(0), state.cx.unsqueeze(0))
    for lstm_param, custom_param in zip(lstm.all_weights[0], rnn.parameters()):
        assert lstm_param.shape == custom_param.shape
        with torch.no_grad():
            lstm_param.copy_(custom_param)
    lstm_out, lstm_out_state = lstm(inp, lstm_state)

    assert (out - lstm_out).abs().max() < 1e-5
    assert (out_state[0] - lstm_out_state[0]).abs().max() < 1e-5
    assert (out_state[1] - lstm_out_state[1]).abs().max() < 1e-5

def test_script_pnr_layer(seq_len, batch, input_size, hidden_size):
    inp = torch.randint(0,input_size,(seq_len, batch))
    inp = nn.functional.one_hot(inp, num_classes=input_size).float()
    state = torch.randn(batch, hidden_size)
    rnn = PNRCell(input_size, hidden_size, polynomial_order=3)
    out,  = rnn(inp[0], (state,))
    num_params = sum(param.numel() for param in rnn.parameters())
    print('Net num_params', num_params)
    print(state, out, )

def test_script_migru_layer(seq_len, batch, input_size, hidden_size):
    inp = torch.randint(0,input_size,(seq_len, batch))
    inp = nn.functional.one_hot(inp, num_classes=input_size).float()
    state = torch.randn(batch, hidden_size)
    rnn = MIGRUCell(input_size, hidden_size)
    out,  = rnn(inp[0], (state,))
    print(out, )


def test_script_gruswitch_layer(seq_len, batch, input_size, hidden_size):
    inp = torch.randint(0,input_size,(seq_len, batch))
    inp = nn.functional.one_hot(inp, num_classes=input_size)
    state = torch.randn(batch, hidden_size)
    rnn = SGRUCell(input_size, hidden_size)
    out,  = rnn(inp[0], (state,))
    print(out, )

def test_script_gru_layer(seq_len, batch, input_size, hidden_size):
    inp = torch.randn(seq_len, batch, input_size)
    state = torch.randn(batch, hidden_size)
    rnn = RNNLayer_custom(input_size, hidden_size)
    out, out_state = rnn(inp, state)

    # Control: pytorch native LSTM
    gru = nn.GRU(input_size, hidden_size, 1)
    gru_state = state.unsqueeze(0)
    for gru_param, custom_param in zip(gru.all_weights[0], rnn.parameters()):
        assert gru_param.shape == custom_param.shape
        with torch.no_grad():
            gru_param.copy_(custom_param)
    gru_out, gru_out_state = gru(inp, gru_state)
    assert (out - gru_out).abs().max() < 1e-5, out
    assert (out_state - gru_out_state).abs().max() < 1e-5, out_state

def test_script_stacked_rnn(seq_len, batch, input_size, hidden_size,
                            num_layers):
    inp = torch.randn(seq_len, batch, input_size)
    states = [LSTMState(torch.randn(batch, hidden_size),
                        torch.randn(batch, hidden_size))
              for _ in range(num_layers)]
    rnn = script_lstm(input_size, hidden_size, num_layers)
    out, out_state = rnn(inp, states)
    custom_state = flatten_states(out_state)

    # Control: pytorch native LSTM
    lstm = nn.LSTM(input_size, hidden_size, num_layers)
    lstm_state = flatten_states(states)
    for layer in range(num_layers):
        custom_params = list(rnn.parameters())[4 * layer: 4 * (layer + 1)]
        for lstm_param, custom_param in zip(lstm.all_weights[layer],
                                            custom_params):
            assert lstm_param.shape == custom_param.shape
            with torch.no_grad():
                lstm_param.copy_(custom_param)
    lstm_out, lstm_out_state = lstm(inp, lstm_state)

    assert (out - lstm_out).abs().max() < 1e-5
    assert (custom_state[0] - lstm_out_state[0]).abs().max() < 1e-5
    assert (custom_state[1] - lstm_out_state[1]).abs().max() < 1e-5


def test_script_stacked_bidir_rnn(seq_len, batch, input_size, hidden_size,
                                  num_layers):
    inp = torch.randn(seq_len, batch, input_size)
    states = [[LSTMState(torch.randn(batch, hidden_size),
                         torch.randn(batch, hidden_size))
               for _ in range(2)]
              for _ in range(num_layers)]
    rnn = script_lstm(input_size, hidden_size, num_layers, bidirectional=True)
    out, out_state = rnn(inp, states)
    custom_state = double_flatten_states(out_state)

    # Control: pytorch native LSTM
    lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True)
    lstm_state = double_flatten_states(states)
    for layer in range(num_layers):
        for direct in range(2):
            index = 2 * layer + direct
            custom_params = list(rnn.parameters())[4 * index: 4 * index + 4]
            for lstm_param, custom_param in zip(lstm.all_weights[index],
                                                custom_params):
                assert lstm_param.shape == custom_param.shape
                with torch.no_grad():
                    lstm_param.copy_(custom_param)
    lstm_out, lstm_out_state = lstm(inp, lstm_state)

    assert (out - lstm_out).abs().max() < 1e-5
    assert (custom_state[0] - lstm_out_state[0]).abs().max() < 1e-5
    assert (custom_state[1] - lstm_out_state[1]).abs().max() < 1e-5


def test_script_stacked_lstm_dropout(seq_len, batch, input_size, hidden_size,
                                     num_layers):
    inp = torch.randn(seq_len, batch, input_size)
    states = [LSTMState(torch.randn(batch, hidden_size),
                        torch.randn(batch, hidden_size))
              for _ in range(num_layers)]
    rnn = script_lstm(input_size, hidden_size, num_layers, dropout=True)

    # just a smoke test
    out, out_state = rnn(inp, states)


def test_script_stacked_lnlstm(seq_len, batch, input_size, hidden_size,
                               num_layers):
    inp = torch.randn(seq_len, batch, input_size)
    states = [LSTMState(torch.randn(batch, hidden_size),
                        torch.randn(batch, hidden_size))
              for _ in range(num_layers)]
    rnn = script_lnlstm(input_size, hidden_size, num_layers)

    # just a smoke test
    out, out_state = rnn(inp, states)


if __name__ == '__main__':
    test_script_pnr_layer(5, 2, 4, 2)
    test_script_migru_layer(5, 2, 4, 3)
    test_script_gruswitch_layer(5, 2, 4, 3)
    test_script_rnn_layer(5, 2, 3, 7)
    test_script_stacked_rnn(5, 2, 3, 7, 4)
    test_script_stacked_bidir_rnn(5, 2, 3, 7, 4)
    test_script_stacked_lstm_dropout(5, 2, 3, 7, 4)
    test_script_stacked_lnlstm(5, 2, 3, 7, 4)
