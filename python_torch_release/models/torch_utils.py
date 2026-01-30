from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

NeuroCoreGuesses = namedtuple('NeuroCoreGuesses', ['pi_core_var_logits'])

PASATGuesses = namedtuple('PASATGuesses',
                                ['pi_var_logits',
                                 'pi_assign_logits',
                                 'var_s', 'var_d', 'var_s_tilde', 'var_d_tilde',
                                 'gate_reg'])


def normalize(x, axis, eps):
    mean = x.mean(dim=axis, keepdim=True)
    var = x.var(dim=axis, unbiased=False, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps)


class MLP(nn.Module):
    def __init__(self, cfg, num_layers, input_dim, hidden_dim, output_dim, activation):
        super(MLP, self).__init__()
        self.cfg = cfg
        self.linear_or_not = True
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("Number of layers should be positive!")
        elif num_layers == 1:
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            l = nn.Linear(input_dim, hidden_dim)
            if cfg['weight_reparam']:
                l = weight_norm(l)
            self.linears.append(l)

            for layer in range(num_layers - 2):
                l = nn.Linear(hidden_dim, hidden_dim)
                if cfg['weight_reparam']:
                    l = weight_norm(l)
                self.linears.append(l)

            self.linears.append(nn.Linear(hidden_dim, output_dim))

        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu6':
            self.activation = F.relu6
        elif activation == None:
            self.activation = lambda x: x
        else:
            raise NotImplementedError("Activation function is not supported!")

    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        else:
            h = x
            for layer in range(self.num_layers - 1):
                h = self.activation(self.linears[layer](h))
            return self.linears[self.num_layers - 1](h)


def check_numerics(x, message=""):
    if torch.isnan(x).any():
        raise RuntimeError(f"{message} contains NaN")
    if torch.isinf(x).any():
        raise RuntimeError(f"{message} contains Inf")
    return x
