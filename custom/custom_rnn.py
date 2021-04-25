"""Implementation of batch-normalized LSTM."""
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional, init
import torch.nn.functional as F
import torch.distributions as tdist
import math
import numpy as np

from custom_utils import to_var

class GumbelNoise(nn.Module):

    def __init__(self, p=0.0, t=1.0, eps=1e-3, noise_type='new_U_B'):
        super(GumbelNoise, self).__init__()
        self.p = p
        self.t = t
        self.eps = eps
        self.noise_type = noise_type
        self.U = None
        self.B = None
        self.noise = None

    def update_noise(self, input_):
        if not self.training or self.p == 0.0:
            return
        if self.noise_type == 'new_U_B':
            pass
        elif self.noise_type == 'new_U':
            self.B = input_.data.new(input_.size()).bernoulli_(self.p)
        elif self.noise_type == 'no_new':
            self.U = input_.data.new(input_.size()).uniform_()
            self.U = torch.log(self.U + self.eps) - torch.log(1 + self.eps - self.U)
            self.B = input_.data.new(input_.size()).bernoulli_(self.p)
            self.noise = self.U * self.B
        else:
            raise ValueError('Unknown noise_type', self.noise_type)

    def forward(self, input_):
        if not self.training or self.p == 0.0:
            return input_
        if self.noise_type == 'new_U_B':
            self.U = input_.data.new(input_.size()).uniform_()
            self.U = torch.log(self.U + self.eps) - torch.log(1 + self.eps - self.U)
            self.B = input_.data.new(input_.size()).bernoulli_(self.p)
            self.noise = self.U * self.B
        elif self.noise_type == 'new_U':
            self.U = input_.data.new(input_.size()).uniform_()
            self.U = torch.log(self.U + self.eps) - torch.log(1 + self.eps - self.U)
            self.noise = self.U * self.B
        elif self.noise_type == 'no_new':
            pass
        else:
            raise ValueError('Unknown noise_type', self.noise_type)
        return (input_ + Variable(self.noise, requires_grad=False)) * (1/self.t)

######################################################
######################################################

class LSTMCell(nn.Module):

    """A basic LSTM cell."""

    def __init__(self, input_size, hidden_size, use_bias=True):
        """
        Most parts are copied from torch.nn.LSTMCell.
        """

        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.weight_ih = nn.Parameter(
            torch.Tensor(4 * hidden_size,input_size))
        self.weight_hh = nn.Parameter(
            torch.Tensor(4 * hidden_size,hidden_size))
        self.weight_hh_wdrop = None
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters following the way proposed in the paper.
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input_, hx, update_noise=True):
        """
        Args:
            input_: A (batch, input_size) tensor containing input
                features.
            hx: A tuple (h_0, c_0), which contains the initial hidden
                and cell state, where the size of both states is
                (batch, hidden_size).
        Returns:
            h_1, c_1: Tensors containing the next hidden and cell state.
        """

        h_0, c_0 = hx
        new_w_hh = self.weight_hh_wdrop \
            if self.weight_hh_wdrop is not None else self.weight_hh
        gates = F.linear(input_, self.weight_ih, self.bias) + F.linear(h_0, new_w_hh)
        f, i, o, g = gates.chunk(4, 1)
        # batch_size = h_0.size(0)
        # bias_batch = (self.bias.unsqueeze(0)
        #               .expand(batch_size, *self.bias.size()))
        # wh_b = torch.addmm(bias_batch, h_0, new_w_hh)
        # wi = torch.mm(input_, self.weight_ih)
        # f, i, o, g = torch.split(wh_b + wi,
        #                          split_size_or_sections=self.hidden_size, dim=1)

        sigm_i = torch.sigmoid(i)
        sigm_f = torch.sigmoid(f)
        gate_value = [sigm_i, sigm_f]
        c_1 = sigm_f*c_0 + sigm_i*torch.tanh(g)
        h_1 = torch.sigmoid(o) * torch.tanh(c_1)
        return h_1, c_1, gate_value, None

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

######################################################
######################################################

class FLSTMCell(nn.Module):

    """A basic LSTM cell."""

    def __init__(self, input_size, hidden_size, use_bias=True):
        """
        Most parts are copied from torch.nn.LSTMCell.
        """

        super(FLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.weight_ih = nn.Parameter(
            torch.Tensor(3 * hidden_size,input_size))
        self.weight_hh = nn.Parameter(
            torch.Tensor(3 * hidden_size,hidden_size))
        self.weight_hh_wdrop = None
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(3 * hidden_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters following the way proposed in the paper.
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input_, hx, update_noise=True):
        """
        Args:
            input_: A (batch, input_size) tensor containing input
                features.
            hx: A tuple (h_0, c_0), which contains the initial hidden
                and cell state, where the size of both states is
                (batch, hidden_size).
        Returns:
            h_1, c_1: Tensors containing the next hidden and cell state.
        """

        h_0, c_0 = hx
        new_w_hh = self.weight_hh_wdrop \
            if self.weight_hh_wdrop is not None else self.weight_hh
        gates = F.linear(input_, self.weight_ih, self.bias) + F.linear(h_0, new_w_hh)
        i, o, g = gates.chunk(3, 1)

        sigm_i = torch.sigmoid(i)
        sigm_f = 1.0 - sigm_i
        #sigm_f = torch.sigmoid(f)
        gate_value = [sigm_i, sigm_f]
        c_1 = sigm_f*c_0 + sigm_i*torch.tanh(g)
        h_1 = torch.sigmoid(o) * torch.tanh(c_1)
        return h_1, c_1, gate_value, None

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

######################################################
######################################################

class NoisinCell(nn.Module):

    """A basic LSTM cell."""

    def __init__(self, input_size, hidden_size, use_bias=True,noisin_noise_type=None,noisin_noise_parama=0.0,noisin_noise_paramb=0.0):
        """
        Most parts are copied from torch.nn.LSTMCell.
        """

        super(NoisinCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.weight_ih = nn.Parameter(
            torch.Tensor(4 * hidden_size,input_size))
        self.weight_hh = nn.Parameter(
            torch.Tensor(4 * hidden_size,hidden_size))
        self.weight_hh_wdrop = None
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.noisin_noise_type = noisin_noise_type
        self.noisin_noise_parama = noisin_noise_parama
        self.noisin_noise_paramb = noisin_noise_paramb
        self.bernoulli_dist = tdist.Bernoulli(noisin_noise_parama)
        self.gamma_dist = tdist.Gamma(noisin_noise_parama,noisin_noise_paramb) # alpha, gammma

    def reset_parameters(self):
        """
        Initialize parameters following the way proposed in the paper.
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def make_bernoulli_noise(self,target_shape):
        eta = to_var(self.bernoulli_dist.sample(sample_shape=target_shape))
        noise = eta * (1/self.noisin_noise_parama)
        return noise

    def make_gamma_noise(self,target_shape):
        eta = to_var(self.bernoulli_dist.sample(sample_shape=target_shape))
        noise = eta - self.noisin_noise_parama * self.noisin_noise_paramb
        noise = noise * (1.0/torch.sqrt(self.noisin_noise_parama)) + 1.0
        return noise

    def forward(self, input_, hx, update_noise=True):
        """
        Args:
            input_: A (batch, input_size) tensor containing input
                features.
            hx: A tuple (h_0, c_0), which contains the initial hidden
                and cell state, where the size of both states is
                (batch, hidden_size).
        Returns:
            h_1, c_1: Tensors containing the next hidden and cell state.
        """

        h_0, c_0 = hx
        new_w_hh = self.weight_hh_wdrop \
            if self.weight_hh_wdrop is not None else self.weight_hh
        gates = F.linear(input_, self.weight_ih, self.bias) + F.linear(h_0, new_w_hh)
        f, i, o, g = gates.chunk(4, 1)

        sigm_i = torch.sigmoid(i)
        sigm_f = torch.sigmoid(f)
        gate_value = [sigm_i, sigm_f]
        c_1 = sigm_f*c_0 + sigm_i*torch.tanh(g)
        if self.training:
            if self.noisin_noise_type == "bernoulli":
                noise = self.make_bernoulli_noise(c_1.size())
            elif self.noisin_noise_type == "gamma":
                noise = self.make_gamma_noise(c_1.size())
            h_1 = torch.sigmoid(o) * torch.tanh(c_1) * noise
        else:
            h_1 = torch.sigmoid(o) * torch.tanh(c_1)
        return h_1, c_1, gate_value, None

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

######################################################
######################################################

class G2LSTMCell(nn.Module):

    """A basic LSTM cell."""

    def __init__(self, input_size, hidden_size, use_bias=True,
                 gumbel_noise_p=0.0, gumbel_noise_t=1.0, gumbel_noise_type='new_U_B',
                 divide_temp=None):
        """
        Most parts are copied from torch.nn.LSTMCell.
        """

        super(G2LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.weight_ih = nn.Parameter(
            torch.Tensor(4 * hidden_size,input_size))
        self.weight_hh = nn.Parameter(
            torch.Tensor(4 * hidden_size,hidden_size))
        self.weight_hh_wdrop = None
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('bias', None)
        self.noisef = GumbelNoise(p=gumbel_noise_p, t=gumbel_noise_t,
                                  noise_type=gumbel_noise_type)
        self.noisei = GumbelNoise(p=gumbel_noise_p, t=gumbel_noise_t,
                                  noise_type=gumbel_noise_type)
        self.divide_temp = divide_temp
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters following the way proposed in the paper.
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

        # init.orthogonal(self.weight_ih.data)
        # weight_hh_data = torch.eye(self.hidden_size)
        # weight_hh_data = weight_hh_data.repeat(1, 4)
        # self.weight_hh.data.set_(weight_hh_data)
        # # The bias is just set to zero vectors.
        # if self.use_bias:
        #     init.constant(self.bias.data, val=0)

    def forward(self, input_, hx, update_noise=True):
        """
        Args:
            input_: A (batch, input_size) tensor containing input
                features.
            hx: A tuple (h_0, c_0), which contains the initial hidden
                and cell state, where the size of both states is
                (batch, hidden_size).
        Returns:
            h_1, c_1: Tensors containing the next hidden and cell state.
        """

        h_0, c_0 = hx
        new_w_hh = self.weight_hh_wdrop \
            if self.weight_hh_wdrop is not None else self.weight_hh
        gates = F.linear(input_, self.weight_ih, self.bias) + F.linear(h_0, new_w_hh)
        f, i, o, g = gates.chunk(4, 1)

        if hasattr(self, 'noisef') and hasattr(self, 'noisei'):
            if update_noise:
                self.noisef.update_noise(f)
                self.noisei.update_noise(i)
            f = self.noisef(f)
            i = self.noisei(i)

        if getattr(self, 'divide_temp', None) is not None:
            f = f * (1.0 / self.divide_temp)
            i = i * (1.0 / self.divide_temp)
        sigm_i = torch.sigmoid(i)
        sigm_f = torch.sigmoid(f)
        # print("==========")
        # print(sigm_i.size())
        # print(sigm_i[0,0:6])
        # print(sigm_f[0,0:6])
        gate_value = [sigm_i, sigm_f]
        c_1 = sigm_f*c_0 + sigm_i*torch.tanh(g)
        h_1 = torch.sigmoid(o) * torch.tanh(c_1)
        return h_1, c_1, gate_value, None

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

######################################################
######################################################

class iBetaLSTMCell(nn.Module):

    """A basic LSTM cell."""

    def __init__(self, input_size, hidden_size, use_bias=True,
                 eps=1e-3):
        """
        Most parts are copied from torch.nn.LSTMCell.
        """

        super(iBetaLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.weight_ih = nn.Parameter(
            torch.Tensor(6 * hidden_size,input_size))
        self.weight_hh = nn.Parameter(
            torch.Tensor(6 * hidden_size,hidden_size))
        self.weight_hh_wdrop = None
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(6 * hidden_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.eps = eps

    def reset_parameters(self):
        """
        Initialize parameters following the way proposed in the paper.
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input_, hx, update_noise=True):
        """
        Args:
            input_: A (batch, input_size) tensor containing input
                features.
            hx: A tuple (h_0, c_0), which contains the initial hidden
                and cell state, where the size of both states is
                (batch, hidden_size).
        Returns:
            h_1, c_1: Tensors containing the next hidden and cell state.
        """

        h_0, c_0 = hx
        new_w_hh = self.weight_hh_wdrop \
            if self.weight_hh_wdrop is not None else self.weight_hh
        gates = F.linear(input_, self.weight_ih, self.bias) + F.linear(h_0, new_w_hh)

        alpha, o, g = torch.split(gates, [self.hidden_size * 4, self.hidden_size, self.hidden_size], dim=1)
        alpha = F.softplus(alpha) + self.eps
        if self.training:
            G = to_var(tdist.Gamma(alpha,1.0).rsample()) + self.eps
            G0,G1,G2,G3 = G.chunk(4,1)
        else:
            G0,G1,G2,G3 = alpha.chunk(4,1) # Expectation

        a_i, b_i = G0, G1
        a_f, b_f = G2, G3
        sigm_i = a_i * (1.0 / (a_i + b_i))
        sigm_f = a_f * (1.0 / (a_f + b_f))  # 1 - G2 * (1/(G2+G0))

        c_1 = sigm_f * c_0 + sigm_i * torch.tanh(g)
        h_1 = torch.sigmoid(o) * torch.tanh(c_1)
        gate_value = [a_i, b_i, a_f, b_f]
        return h_1, c_1, gate_value,None

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

######################################################
######################################################

class bBetaLSTMCell(nn.Module):

    """A basic LSTM cell."""

    def __init__(self, input_size, hidden_size, use_bias=True,
                 eps=1e-3):
        """
        Most parts are copied from torch.nn.LSTMCell.
        """

        super(bBetaLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.weight_ih = nn.Parameter(
            torch.Tensor(5 * hidden_size,input_size))
        self.weight_hh = nn.Parameter(
            torch.Tensor(5 * hidden_size,hidden_size))
        self.weight_hh_wdrop = None
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(5 * hidden_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.eps = eps

    def reset_parameters(self):
        """
        Initialize parameters following the way proposed in the paper.
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input_, hx, update_noise=True):
        """
        Args:
            input_: A (batch, input_size) tensor containing input
                features.
            hx: A tuple (h_0, c_0), which contains the initial hidden
                and cell state, where the size of both states is
                (batch, hidden_size).
        Returns:
            h_1, c_1: Tensors containing the next hidden and cell state.
        """

        h_0, c_0 = hx
        new_w_hh = self.weight_hh_wdrop \
            if self.weight_hh_wdrop is not None else self.weight_hh
        gates = F.linear(input_, self.weight_ih, self.bias) + F.linear(h_0, new_w_hh)

        alpha, o, g = torch.split(gates, [self.hidden_size * 3, self.hidden_size, self.hidden_size], dim=1)
        alpha = F.softplus(alpha) + self.eps
        if self.training:
            G = to_var(tdist.Gamma(alpha,1.0).rsample()) + self.eps
            G0,G1,G2 = G.chunk(3,1)
        else:
            G0,G1,G2 = alpha.chunk(3,1) # Expectation

        a_i,b_i = G1, G2
        a_f,b_f = G0, G2
        ### Check for sigm_f ###
        sigm_i = a_i * (1.0/(a_i+b_i)) # G1 * (1/(G1+G2))
        sigm_f = a_f * (1.0/(a_f+b_f)) # positive corr
        # sigm_f = b_f * (1.0 / (a_f + b_f))  # negative corr

        c_1 = sigm_f*c_0 + sigm_i*torch.tanh(g)
        h_1 = torch.sigmoid(o) * torch.tanh(c_1)
        gate_value = [a_i,b_i,a_f,b_f]
        return h_1, c_1,gate_value,None


    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

######################################################
######################################################

class gBetaLSTMCell(nn.Module):

    """A basic LSTM cell."""

    def __init__(self, input_size, hidden_size, use_bias=True,
                 eps=1e-3,kl_gamma_prior=None):
        """
        Most parts are copied from torch.nn.LSTMCell.
        """

        super(gBetaLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.weight_ih = nn.Parameter(
            torch.Tensor(7 * hidden_size,input_size))
        self.weight_hh = nn.Parameter(
            torch.Tensor(7 * hidden_size,hidden_size))
        self.weight_hh_wdrop = None
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(7 * hidden_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.eps = eps
        self.kl_gamma_prior = kl_gamma_prior

    def reset_parameters(self):
        """
        Initialize parameters following the way proposed in the paper.
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input_, hx, update_noise=True):
        """
        Args:
            input_: A (batch, input_size) tensor containing input
                features.
            hx: A tuple (h_0, c_0), which contains the initial hidden
                and cell state, where the size of both states is
                (batch, hidden_size).
        Returns:
            h_1, c_1: Tensors containing the next hidden and cell state.
        """

        h_0, c_0 = hx
        new_w_hh = self.weight_hh_wdrop \
            if self.weight_hh_wdrop is not None else self.weight_hh
        gates = F.linear(input_, self.weight_ih, self.bias) + F.linear(h_0, new_w_hh)

        alpha, o, g = torch.split(gates, [self.hidden_size * 5, self.hidden_size, self.hidden_size], dim=1)
        alpha = F.softplus(alpha) + self.eps
        if self.training:
            G = to_var(tdist.Gamma(alpha,1.0).rsample()) + self.eps
            G1,G2,G3,G4,G5 = G.chunk(5,1)
        else:
            G1,G2,G3,G4,G5 = alpha.chunk(5,1) # Expectation

        a_i,b_i = G1 + G3, G4 + G5
        a_f,b_f = G2 + G4, G3 + G5
        sigm_i = a_i * (1.0/(a_i+b_i))
        sigm_f = a_f * (1.0/(a_f+b_f)) # 1 - G2 * (1/(G2+G0))

        c_1 = sigm_f*c_0 + sigm_i*torch.tanh(g)
        h_1 = torch.sigmoid(o) * torch.tanh(c_1)
        gate_value = [a_i,b_i,a_f,b_f]

        q = tdist.Gamma(alpha,1.0)
        p = tdist.Gamma(self.kl_gamma_prior,1.0)
        kl_loss = torch.distributions.kl.kl_divergence(q,p).sum(dim=1)

        return h_1, c_1, gate_value, kl_loss

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

######################################################
######################################################

class LSTM(nn.Module):

    """A module that runs multiple steps of LSTM."""

    def __init__(self, args,cell_class, input_size, hidden_size, num_layers=1,
                 use_bias=True, batch_first=False, dropout=0, wdrop=None, **kwargs):
        super(LSTM, self).__init__()
        self.cell_class = cell_class
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_bias = use_bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.wdrop = wdrop

        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            cell = cell_class(input_size=layer_input_size,
                              hidden_size=hidden_size,
                              **kwargs)
            setattr(self, 'cell_{}'.format(layer), cell)
        self.dropout_layer = nn.Dropout(dropout)
        self.reset_parameters()

    def get_cell(self, layer):
        return getattr(self, 'cell_{}'.format(layer))

    def reset_parameters(self):
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            cell.reset_parameters()

    def reset_gumbel_noise(self, gumbel_noise_p=0.0, gumbel_noise_t=1.0,
                           gumbel_noise_type='new_U_B'):
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)

            if not hasattr(cell, 'noisef'):
                cell.noisef = GumbelNoise(p=gumbel_noise_p, t=gumbel_noise_t,
                                          noise_type=gumbel_noise_type)
            else:
                cell.noisef.p = gumbel_noise_p
                cell.noisef.t = gumbel_noise_t
                cell.noisef.noise_type = gumbel_noise_type

            if not hasattr(cell, 'noisei'):
                cell.noisei = GumbelNoise(p=gumbel_noise_p, t=gumbel_noise_t,
                                          noise_type=gumbel_noise_type)
            else:
                cell.noisei.p = gumbel_noise_p
                cell.noisei.t = gumbel_noise_t
                cell.noisei.noise_type = gumbel_noise_type

    @staticmethod
    def _forward_rnn(cell, input_, hx):
        max_time = input_.size(0)
        output = []
        gate = []
        kl_loss = []
        for time in range(max_time):
            h_next, c_next,gate_next,kl_loss_next = cell(input_=input_[time], hx=hx,
                                  update_noise=(time == 0))
            hx_next = (h_next, c_next)
            output.append(h_next)
            gate.append(gate_next)
            kl_loss.append(kl_loss_next)
            hx = hx_next
        output = torch.stack(output, 0)
        try:
            kl_loss = torch.stack(kl_loss, 0)
        except:
            kl_loss = None
        return output, hx, gate,kl_loss

    def forward(self, input_, hx=None):
        if self.batch_first:
            input_ = input_.transpose(0, 1)

        max_time, batch_size, _ = input_.size()

        if hx is None:
            hx = Variable(input_.data.new(batch_size, self.hidden_size).zero_())
            hx = [(hx, hx) for _ in range(self.num_layers)]
        layer_output = None
        new_hx = []
        for layer in range(self.num_layers):
            global global_layer
            global_layer = layer
            cell = self.get_cell(layer)
            # if self.wdrop is not None:
            if self.wdrop>0:
                cell.weight_hh_wdrop = torch.nn.Parameter(torch.nn.functional.dropout(
                    cell.weight_hh, self.wdrop, training=self.training))
            layer_output, (layer_h_n, layer_c_n),gate,kl_loss = LSTM._forward_rnn(
                cell=cell, input_=input_, hx=hx[layer])
            input_ = self.dropout_layer(layer_output)
            new_hx.append((layer_h_n, layer_c_n))
        output = layer_output
        return output, new_hx, gate,kl_loss