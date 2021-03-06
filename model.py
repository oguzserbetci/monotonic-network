import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor


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
            pos_lin = PositiveLinear(n_inputs, n_units, bias=True)
            self.add_module(f'group_{i}', pos_lin)

        self.groups = AttrProxy(self, 'group_', self.n_groups)
        self.group_activations = []
        self.hp_activations = []

    def forward(self, *args):
        x = torch.stack([FloatTensor(arg) for arg in args],1)

        us = []
        for g in self.groups:
            us.append(g(x))

        m, active_hp = torch.max(torch.stack(us), -1)
        self.hp_activations.append(active_hp)

        y, active_group = torch.min(m, 0, keepdim=True)
        self.group_activations.append(active_group)
        return y.t()


class PositiveLinear(nn.Linear):
    def forward(self, input):
        return F.linear(input, self.weight**2, self.bias)
