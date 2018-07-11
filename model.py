import torch
import torch.nn as nn
import torch.nn.functional as F


class AttrProxy(object):
    """Translates index lookups into attribute lookups."""
    def __init__(self, module, prefix, n_modules):
        self.module = module
        self.prefix = prefix
        self.n_modules = n_modules

    def __len__(self):
        return self.n_modules

    def __getitem__(self, i):
        if i < self.__len__():
            return getattr(self.module, self.prefix + str(i))
        else:
            raise IndexError


class MonotonicNetwork(nn.Module):

    def __init__(self, n_inputs, groups=[3,3,3]):
        nn.Module.__init__(self)

        self.n_groups = len(groups)
        for i, n_units in enumerate(groups):
            self.add_module(f'group_{i}', PositiveLinear(n_inputs, n_units, bias=True))

        self.groups = AttrProxy(self, 'group_', self.n_groups)
        self.group_activations = []
        self.hp_activations = []

    def forward(self, x):
        us = []
        for g in self.groups:
            us.append(g(x))

        m, active_hp = torch.max(torch.stack(us), -1)
        self.hp_activations.append(active_hp)

        y, active_group = torch.min(m, 0, keepdim=True)
        self.group_activations.append(active_group)
        return y.t()

class PositiveLinear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        nn.Module.__init__(self)
        self.input_features = input_features
        self.output_features = output_features

        weight = torch.Tensor(output_features, input_features)
        init_weight = nn.init.xavier_normal_(weight)
        self.weight = nn.Parameter(init_weight)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
            self.bias.data.uniform_(-0.1, 0.1)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        return F.linear(input, self.weight**2, self.bias)
