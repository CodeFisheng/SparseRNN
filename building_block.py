"""Implementation of Spasified hidden LSTM."""
import math
import torch
from torch import nn
# from torch.autograd import Variable
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
            return grad_output.mul(ctx.mask), None, None, None, None
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
    # mask = Variable(x.data, requires_grad=False)
    mask = x.detach()
    mask = mask.abs()

    # for sample_idx in range(m):
    # approximately choose the top-k based on the first sample
    mask_topk = torch.topk(mask.view(-1), k, sorted=False)
    threshold = float(torch.min(mask_topk[0]))
    mask = F.threshold(mask, threshold, 0)
    # mask = mask.abs()
    mask = torch.sign(mask)

    return mask


class LSTMCell(nn.Module):

    """A basic LSTM cell."""

    def __init__(self, input_size, hidden_size, bias=True, sparsity_ratio=None):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
            self.bias_hh = nn.Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()
        self.sparsity_ratio = sparsity_ratio

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def forward(self, input, hx=None):
        # self.check_forward_input(input)
        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
            hx = (hx, hx)
        # self.check_forward_hidden(input, hx[0], '[0]')
        # self.check_forward_hidden(input, hx[1], '[1]')
        return self.LSTMCell_func(
            input, hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh,
        )

    def LSTMCell_func(self, input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
        hx, cx = hidden
        gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        # modified by Liu
        if self.sparsity_ratio:
            mask = topk_mask_gen(gates, 1 - self.sparsity_ratio)
            assert mask.size() == gates.size()
            gates.data.mul_(mask)
            # gates = gates * mask
            # print('before mask: ', gates.nonzero().size(0))
            # gates = my_mask(gates, mask, training=True, inplace=True)
            # print('after  mask: ', net.nonzero().size(0))


        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, cy

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

        self.profiling = True

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
            h_next, c_next = cell(input=input_[time], hx=hx)
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
            length = torch.LongTensor([max_time] * batch_size)
            if input_.is_cuda:
                device = input_.get_device()
                length = length.cuda(device)
        if hx is None:
            hx = input_.new_zeros(batch_size, self.hidden_size)
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

        if self.profiling:
        	print('input: ', input_.size())
        	print('hx, hx: ', hx[0].size(), hx[1].size())
        	print('output: ', output.size())
        	print('h_n, c_n: ', h_n.size(), c_n.size())
        	self.profiling = False

        return output, (h_n, c_n)