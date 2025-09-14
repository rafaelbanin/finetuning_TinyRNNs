"""Customized rnn layers.

Mainly GRU, MIGRU, SGRU, PNR, etc.
In a nn.Module way, instead of a TorchScript way.
"""
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch import Tensor
import numbers
import math
import torch.jit as jit

class GRUCell(nn.Module):
    """One-step GRU cell, same as the official version."""
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(3 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(3 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(3 * hidden_size))
        self.bias_hh = Parameter(torch.randn(3 * hidden_size))

    def forward(self, input, state):
        hx = state[0, ...] # ignore first dim, seq_len=1
        gates_i = torch.mm(input, self.weight_ih.t()) + self.bias_ih
        gates_h = torch.mm(hx, self.weight_hh.t()) + self.bias_hh
        reset_i, update_i, new_i = gates_i.chunk(3, 1)
        reset_h, update_h, new_h = gates_h.chunk(3, 1)
        reset_gate = torch.sigmoid(reset_i + reset_h)
        update_gate = torch.sigmoid(update_i + update_h)
        new_gate = torch.tanh(new_i + reset_gate * new_h)
        hy = update_gate * hx + (1 - update_gate) * new_gate

        return hy[None, ...] # insert back first dim

class LRCell(nn.Module):
    """One-step low-rank RNN cell."""
    def __init__(self, input_size, hidden_size, rank, nonlinearity='tanh'):
        super(LRCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rank = rank
        self.weight_ih = Parameter(torch.randn(hidden_size, input_size))
        self.weight_hh_m = Parameter(torch.randn(hidden_size, rank))
        self.weight_hh_n = Parameter(torch.randn(rank, hidden_size))
        self.bias = Parameter(torch.randn(hidden_size))
        self.nonlinearity = nonlinearity

    def forward(self, input, state):
        hx = state[0, ...] # ignore first dim, seq_len=1
        weight_hh = torch.mm(self.weight_hh_m, self.weight_hh_n)
        hy = torch.mm(input, self.weight_ih.t()) + torch.mm(hx, weight_hh.t()) + self.bias
        if self.nonlinearity == 'tanh':
            hy = torch.tanh(hy)
        elif self.nonlinearity == 'relu':
            hy = torch.relu(hy)
        else:
            raise NotImplementedError
        return hy[None, ...] # insert back first dim

# class MLRCell(nn.Module):
class MLRCell(jit.ScriptModule):
    """One-step modified low-rank RNN cell."""
    def __init__(self, input_size, hidden_size, expand_size, nonlinearity='tanh'):
        super(MLRCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size # bottleneck size
        self.expand_size = expand_size # expand size
        self.weight_ih = Parameter(torch.randn(expand_size, input_size)/100)
        self.weight_hh_m = Parameter(torch.randn(expand_size, hidden_size)/100)
        self.weight_hh_n = Parameter(torch.randn(hidden_size, expand_size)/100)
        self.bias = Parameter(torch.randn(expand_size)/100)
        self.nonlinearity = nonlinearity

    def reset_parameters(self):
        for weight in self.parameters():
            # WARNING: this is not the same as the init distribution
            weight.data.uniform_(-0.01, 0.01)

    @jit.script_method
    def forward(self, input, state):
        hx = state[0, ...] # ignore first dim, [seq_len=1, batch, hidden_size]
        hy = torch.mm(input, self.weight_ih.t()) + torch.mm(hx, self.weight_hh_m.t()) + self.bias
        if self.nonlinearity == 'tanh':
            hy = torch.tanh(hy)
        elif self.nonlinearity == 'relu': # relu is less used in rnn
            hy = torch.relu(hy)
        elif self.nonlinearity == 'sigmoid':
            hy = torch.sigmoid(hy)
        elif self.nonlinearity == 'linear': # with linear activation, the network is a linear network without switching
            pass
        else:
            raise NotImplementedError
        hy = torch.mm(hy, self.weight_hh_n.t()) # + hx # residual connection seems not helpful here
        return hy[None, ...] # insert back first dim


from torch.utils.cpp_extension import load
import os
# os.chdir(r'D:\OneDrive\Documents\git_repo\cognitive_dynamics\agents\mlr')
# mlr_cpp = load(name="mlr_cpp", sources=["mlr.cpp"])

class MLRFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, state, weight_ih, weight_hh_m, weight_hh_n, bias):
        output, pre_hy, hy = mlr_cpp.forward(input, state, weight_ih, weight_hh_m, weight_hh_n, bias)
        variables = [input, state, weight_ih, weight_hh_m, weight_hh_n, bias, output, pre_hy, hy]
        ctx.save_for_backward(*variables)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input, grad_state, grad_weight_ih, grad_weight_hh_m, grad_weight_hh_n, grad_bias = mlr_cpp.backward(
            grad_output.contiguous(), *ctx.saved_tensors)
        return grad_input, grad_state, grad_weight_ih, grad_weight_hh_m, grad_weight_hh_n, grad_bias

class MLR(MLRCell):
    def __init__(self, input_size, hidden_size, expand_size):
        super(MLR, self).__init__(input_size, hidden_size, expand_size)

    def forward(self, input, state):
        return MLRFunction.apply(input, state, self.weight_ih, self.weight_hh_m, self.weight_hh_n, self.bias)


class MLRLayerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, initial_state, weight_ih, weight_hh_m, weight_hh_n, bias):
        outputs, state, pre_hys, hys = mlr_cpp.time_forward(input, initial_state, weight_ih, weight_hh_m,
                                                                     weight_hh_n, bias)

        variables = [input, initial_state, weight_ih, weight_hh_m, weight_hh_n, bias, outputs, pre_hys, hys]
        ctx.save_for_backward(*variables)
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        grad_input, grad_initial_state, grad_weight_ih, grad_weight_hh_m, grad_weight_hh_n, grad_bias = mlr_cpp.time_backward(
            grad_outputs, *ctx.saved_tensors
        )
        return grad_input, grad_initial_state, grad_weight_ih, grad_weight_hh_m, grad_weight_hh_n, grad_bias


class MLRLayer(MLRCell):
    def __init__(self, input_dim, hidden_dim, expand_size):
        super(MLRLayer, self).__init__(input_dim, hidden_dim, expand_size)

    def forward(self, input, state):
        # state is initial state
        return MLRLayerFunction.apply(input, state, self.weight_ih, self.weight_hh_m, self.weight_hh_n,
                                      self.bias)

class MIGRUCell(nn.Module):
    """One-step multiplicative integration GRU cell, same as the original paper."""
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

    def forward(self, input, state):
        hx = state[0] # ignore first dim, seq_len=1
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

        return hy[None, ...] # insert back first dim

class SGRUCell(nn.Module):
    """one-step switching GRU.

    Modification of the original GRU. Because the input is one-hot (for combinations of discrete input features),
    will use different recurrent weight depending on the input.
    """
    def __init__(self, input_size, hidden_size):
        super(SGRUCell, self).__init__()
        self.input_size = input_size # input_size is for one-hot encoding
        self.hidden_size = hidden_size
        #self.weight_ih = Parameter(torch.randn(3 * hidden_size, input_size)) # can use this for continous input dimension
        self.weight_hh = Parameter(torch.randn(3 * hidden_size, hidden_size, input_size)) # 3H*H*I
        self.bias_ih = Parameter(torch.randn(3 * hidden_size, input_size)) # 3H*I
        self.bias_hh = Parameter(torch.randn(3 * hidden_size, input_size)) # 3H*I

    def forward(self, input, state):
        # input is one-hot: batch_size, input_size
        assert (input.sum(-1) == 1).all(), input.sum(-1)
        hx = state[0] # B*H # ignore first dim, seq_len=1
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

        return hy[None, ...] # insert back first dim


class PNRCell(nn.Module):
    """One-step PolyNomial Regression cell.
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


    def forward(self, input, state):
        # input is one-hot: batch_size, input_size
        assert (input.sum(-1) == 1).all()
        trial_weight = (self.weight[None, :,:,:] * input[:, None, None, :]).sum(-1) # 1*H*F*I, B*1*1*I-> B*H*F
        trial_bias = (self.bias[None, :,:] * input[:, None, :]).sum(-1) # 1*H*I, B*1*I-> B*H
        hx = state[0] # B*H # ignore first dim, seq_len=1
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

        return hy[None, ...] # insert back first dim

class SRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SRUCell, self).__init__()
        self.input_size = input_size # input_size is for one-hot
        self.hidden_size = hidden_size
        self.weight = Parameter(torch.zeros(hidden_size, hidden_size, input_size)) # H*F*I
        self.bias = Parameter(torch.zeros(hidden_size, input_size)) # H*I

    def forward(self, input, state):
        # input is one-hot: batch_size, input_size
        assert (input.sum(-1) == 1).all()
        hx = state[0] # B*H # ignore first dim, seq_len=1
        trial_weight = (self.weight[None, :,:,:] * input[:, None, None, :]).sum(-1) # 1*H*F*I, B*1*1*I-> B*H*F
        trial_bias = (self.bias[None, :,:] * input[:, None, :]).sum(-1) # 1*H*I, B*1*I-> B*H
        rec_temp = (trial_weight * hx[:, None, :]).sum(-1) # B*H*F, B*1*F->B*H
        hy = hx + rec_temp + trial_bias
        return hy[None, ...] # insert back first dim [seq_len=1, batch, hidden_size]

class PNRCellSymm(PNRCell):
    """One step Polynomial regression cell, with symmetric weights.
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
        self._weight = Parameter(torch.zeros(hidden_size, feature_size, input_size)) # H*F*I
        self.bias = Parameter(torch.zeros(hidden_size, input_size)) # H*I


    @property
    def weight(self):
        return (self._weight + self._weight.transpose(0, 1)) / 2

# class RNNLayer_custom(nn.Module):
class RNNLayer_custom(jit.ScriptModule):
    """Customized RNN layer.

    Attributes:
        rnn_type:
        rnncell: a cell responsible for a single time step.
    """
    def __init__(self, *cell_args, **kwargs):
        super().__init__()
        self.rnn_type = kwargs['rnn_type']
        if self.rnn_type == 'SGRU':
            self.rnncell = SGRUCell(*cell_args)
        elif self.rnn_type == 'MIGRU':
            self.rnncell = MIGRUCell(*cell_args)
        elif self.rnn_type == 'GRU':
            self.rnncell = GRUCell(*cell_args)
        elif 'PNR' in self.rnn_type:
            if 'symm' in kwargs and kwargs['symm']:
                self.rnncell = PNRCellSymm(*cell_args, polynomial_order=kwargs['polynomial_order'])
            else:
                self.rnncell = PNRCell(*cell_args, polynomial_order=kwargs['polynomial_order'])
        elif 'MLR' in self.rnn_type:
            if 'nonlinearity' in kwargs:
                nonlinearity = kwargs['nonlinearity']
            else:
                nonlinearity = 'tanh'
            self.rnncell = MLRCell(*cell_args, expand_size=kwargs['expand_size'], nonlinearity=nonlinearity)
        elif 'LR' in self.rnn_type:
            if 'nonlinearity' in kwargs:
                nonlinearity = kwargs['nonlinearity']
            else:
                nonlinearity = 'tanh'
            self.rnncell = LRCell(*cell_args, rank=kwargs['rank'], nonlinearity=nonlinearity)

        else:
            print(self.rnn_type)
            raise NotImplementedError

    @jit.script_method
    def forward(self, input, state):
        """Run one-step RNN cell for several time steps.

        Args:
            input: shape: seq_len, batch_size, input_dim
            state: shape: seq_len=1, batch_size, hidden_dim

        Returns:
            outputs: shape: seq_len, batch_size, output_dim
            state: final state after seeing all inputs, shape: seq_len=1, batch_size, hidden_dim
        """
        assert len(state.shape) == 3
        inputs = input.unbind(0)
        outputs = []
        for i in range(len(inputs)):
            state = self.rnncell(inputs[i], state)
            out = state
            outputs += [out]
        assert len(state.shape) == 3
        return torch.cat(outputs, 0), state


def test_mlr(seq_len, batch, input_size, hidden_size, expand_size):
    from copy import deepcopy
    inp = torch.randn(seq_len, batch, input_size, requires_grad=True)
    state = torch.randn(batch, hidden_size, requires_grad=True)
    inp_ori = inp.clone().detach().requires_grad_(True)
    state_ori = state.clone().detach().requires_grad_(True)

    # c version
    rnn_c_full = MLRLayer(input_size, hidden_size, expand_size)
    out_full = rnn_c_full(inp, state)

    # pytorch version
    rnn_ori_full = RNNLayer_custom(input_size, hidden_size, rnn_type='MLR', expand_size=expand_size)
    out_ori_full = rnn_ori_full(inp_ori, state_ori[None, ...])
    return
    rnn_c = MLR(input_size, hidden_size, expand_size)
    out = rnn_c(inp[0], state)
    loss_c = out.sum()
    loss_c.backward()
    inp_grad, state_grad = inp.grad, state.grad
    # print(out, )
    print(inp_grad, state_grad)

    # pytorch version
    rnn_ori = MLRCell(input_size, hidden_size, expand_size)
    for param, param_ori in zip(rnn_c.parameters(), rnn_ori.parameters()):
        assert param.shape == param_ori.shape
        with torch.no_grad():
            param_ori.data = deepcopy(param.data)
    out_ori = rnn_ori(inp_ori[0], state_ori[None, ...])
    loss_ori = out_ori.sum()
    loss_ori.backward()
    inp_grad_ori, state_grad_ori = inp_ori.grad, state_ori.grad
    print(inp_grad_ori, state_grad_ori)
    # print(out_ori)
    assert (out - out_ori).abs().max() < 1e-5, out

    assert (inp_grad - inp_grad_ori).abs().max() < 1e-5, inp_grad
    assert (state_grad - state_grad_ori).abs().max() < 1e-5, state_grad

    for param, param_ori in zip(rnn_c.parameters(), rnn_ori.parameters()):
        # check if the gradient is the same
        print((param.grad - param_ori.grad).abs().max())
        assert (param.grad - param_ori.grad).abs().max() < 1e-5, param.grad

def test_script_pnr_layer(seq_len, batch, input_size, hidden_size):
    inp = torch.randint(0,input_size,(seq_len, batch))
    inp = nn.functional.one_hot(inp, num_classes=input_size).float()
    state = torch.randn(1,batch, hidden_size)
    rnn = PNRCell(input_size, hidden_size, polynomial_order=3)
    out = rnn(inp[0], state)
    num_params = sum(param.numel() for param in rnn.parameters())
    print('Net num_params', num_params)
    print(state, out, )

def test_script_migru_layer(seq_len, batch, input_size, hidden_size):
    inp = torch.randint(0,input_size,(seq_len, batch))
    inp = nn.functional.one_hot(inp, num_classes=input_size).float()
    state = torch.randn(1, batch, hidden_size)
    rnn = MIGRUCell(input_size, hidden_size)
    out  = rnn(inp[0], state)
    print(out, )


def test_script_gruswitch_layer(seq_len, batch, input_size, hidden_size):
    inp = torch.randint(0,input_size,(seq_len, batch))
    inp = nn.functional.one_hot(inp, num_classes=input_size)
    state = torch.randn(1, batch, hidden_size)
    rnn = SGRUCell(input_size, hidden_size)
    out  = rnn(inp[0], state)
    print(out, )

def test_script_gru_layer(seq_len, batch, input_size, hidden_size):
    inp = torch.randn(seq_len, batch, input_size)
    state = torch.randn(1, batch, hidden_size)
    rnn = RNNLayer_custom(input_size, hidden_size, rnn_type='GRU')
    out, out_state = rnn(inp, state)

    # Control: pytorch native LSTM
    gru = nn.GRU(input_size, hidden_size, 1)
    gru_state = state#.unsqueeze(0)
    for gru_param, custom_param in zip(gru.all_weights[0], rnn.parameters()):
        assert gru_param.shape == custom_param.shape
        with torch.no_grad():
            gru_param.copy_(custom_param)
    gru_out, gru_out_state = gru(inp, gru_state)
    assert (out - gru_out).abs().max() < 1e-5, out
    assert (out_state - gru_out_state).abs().max() < 1e-5, out_state


if __name__ == '__main__':
    test_mlr(5, 2, 4, 3, 10)
    # test_script_gru_layer(5, 2, 4, 3)
    # test_script_pnr_layer(5, 2, 4, 2)
    # test_script_migru_layer(5, 2, 4, 3)
    # test_script_gruswitch_layer(5, 2, 4, 3)

