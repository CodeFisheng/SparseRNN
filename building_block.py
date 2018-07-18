"""Implementation of Spasified hidden LSTM."""
import math
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import init
from torch.autograd.function import InplaceFunction
import torch.nn.functional as F

def my_mask(input, mask, p=0.5, training=False, inplace=False):
    # p is the probability of each hidden neuron being dropped
    return MyMaskLayer.apply(input, mask, p, training, inplace)

class MyMaskLayer(InplaceFunction):

    @staticmethod
    def _make_mask(m, p):
        return m.div_(p)

    @classmethod
    def forward(cls, ctx, input, mask, p, train=False, inplace=False):
        assert input.size() == mask.size()
        if p < 0 or p > 1:
            raise ValueError("Drop probability has to be between 0 and 1, "
                            "but got {}".format(p))

        ctx.p = p
        ctx.train = train
        ctx.inplace = inplace

        if ctx.p == 0 or not ctx.train:
            return input

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        ctx.mask = mask
        if ctx.p == 1:
            ctx.mask.fill_(0)
        # print('mask: ', ctx.mask)

        output.mul_(ctx.mask)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.p > 0 and ctx.train:
            return grad_output.mul(Variable(ctx.mask)), None, None, None, None
        else:
            return grad_output, None, None, None, None

def topk_mask_gen(x, top):
    # print(x.size())
    assert x.dim() == 2

    # x: [mini-batch size, hidden size]
    # extract mini-batch size m
    m = x.size(0)
    k = int(top * x.size(0) * x.size(1))
    # k = int(top * x.size(0) * x.size(1) * x.size(2) * x.size(3))
    # print('k: ', k)
    mask = Variable(x.data, requires_grad=False)
    mask = mask.abs()

    # for sample_idx in range(m):
    # approximately choose the top-k based on the first sample
    mask_topk = torch.topk(mask.view(-1), k, sorted=False)
    threshold = float(torch.min(mask_topk[0]))
    mask = F.threshold(mask, threshold, 0)
    # mask = mask.abs()
    mask = torch.sign(mask)

    return mask

class SparseLSTMCell(nn.Module):

    """A basic LSTM cell."""

    def __init__(self, input_size, hidden_size, use_bias=True):
        """
        Most parts are copied from torch.nn.SparseLSTMCell.
        """

        super(SparseLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.weight_ih = nn.Parameter(
            torch.FloatTensor(input_size, 4 * hidden_size))
        self.weight_hh = nn.Parameter(
            torch.FloatTensor(hidden_size, 4 * hidden_size))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(4 * hidden_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.sparsity_ratio = 0.5

    def reset_parameters(self):
        """
        Initialize parameters following the way proposed in 
        Recurrent Batch Normalization.
        """

        # init.orthogonal(self.weight_ih.data)
        # weight_hh_data = torch.eye(self.hidden_size)
        # weight_hh_data = weight_hh_data.repeat(1, 4)
        # self.weight_hh.data.set_(weight_hh_data)
        # # The bias is just set to zero vectors.
        # if self.use_bias:
        #     init.constant(self.bias.data, val=0)
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input_, hx):
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
        batch_size = h_0.size(0)
        bias_batch = (self.bias.unsqueeze(0)
                      .expand(batch_size, *self.bias.size()))
        wh_b = torch.addmm(bias_batch, h_0, self.weight_hh)
        wi = torch.mm(input_, self.weight_ih)
        net = wh_b + wi
        # modified by Liu
        mask = topk_mask_gen(net, 1 - self.sparsity_ratio)
        assert mask.size() == net.size()
        # net = net * mask
        net = my_mask(net, mask, training=True)

        f, i, o, g = torch.split(net, split_size=self.hidden_size, dim=1)
        c_1 = torch.sigmoid(f) * c_0 + torch.sigmoid(i) * torch.tanh(g)
        h_1 = torch.sigmoid(o) * torch.tanh(c_1)
        return h_1, c_1

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


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
            torch.FloatTensor(input_size, 4 * hidden_size))
        self.weight_hh = nn.Parameter(
            torch.FloatTensor(hidden_size, 4 * hidden_size))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(4 * hidden_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters following the way proposed in 
        Recurrent Batch Normalization.
        """

        # init.orthogonal(self.weight_ih.data)
        # weight_hh_data = torch.eye(self.hidden_size)
        # weight_hh_data = weight_hh_data.repeat(1, 4)
        # self.weight_hh.data.set_(weight_hh_data)
        # # The bias is just set to zero vectors.
        # if self.use_bias:
        #     init.constant(self.bias.data, val=0)
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input_, hx):
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
        batch_size = h_0.size(0)
        bias_batch = (self.bias.unsqueeze(0)
                      .expand(batch_size, *self.bias.size()))
        wh_b = torch.addmm(bias_batch, h_0, self.weight_hh)
        wi = torch.mm(input_, self.weight_ih)
        net = wh_b + wi
        f, i, o, g = torch.split(net, split_size=self.hidden_size, dim=1)
        c_1 = torch.sigmoid(f) * c_0 + torch.sigmoid(i) * torch.tanh(g)
        h_1 = torch.sigmoid(o) * torch.tanh(c_1)
        return h_1, c_1

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

class LSTM(nn.Module):

    """A module that runs multiple steps of LSTM."""

    def __init__(self, cell_class, input_size, hidden_size, num_layers=1,
                 use_bias=True, batch_first=False, dropout=0, **kwargs):
        super(LSTM, self).__init__()
        self.cell_class = cell_class
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_bias = use_bias
        self.batch_first = batch_first
        self.dropout = dropout

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

    @staticmethod
    def _forward_rnn(cell, input_, length, hx):
        max_time = input_.size(0)
        output = []
        for time in range(max_time):
            h_next, c_next = cell(input_=input_[time], hx=hx)
            # mask = (time < length).float().unsqueeze(1).expand_as(h_next)
            # h_next = h_next*mask + hx[0]*(1 - mask)
            # c_next = c_next*mask + hx[1]*(1 - mask)
            hx_next = (h_next, c_next)
            output.append(h_next)
            hx = hx_next
        output = torch.stack(output, 0)
        return output, hx

    def forward(self, input_, length=None, hx=None):
        if self.batch_first:
            input_ = input_.transpose(0, 1)
        max_time, batch_size, _ = input_.size()
        if length is None:
            length = Variable(torch.LongTensor([max_time] * batch_size))
            if input_.is_cuda:
                device = input_.get_device()
                length = length.cuda(device)
        if hx is None:
            hx = Variable(input_.data.new(batch_size, self.hidden_size).zero_())
            hx = (hx, hx)
        h_n = []
        c_n = []
        layer_output = None
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            layer_output, (layer_h_n, layer_c_n) = LSTM._forward_rnn(
                cell=cell, input_=input_, length=length, hx=hx)
            input_ = self.dropout_layer(layer_output)
            h_n.append(layer_h_n)
            c_n.append(layer_c_n)
        output = layer_output
        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n, 0)
        return output, (h_n, c_n)