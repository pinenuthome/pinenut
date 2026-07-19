"""Neural-network modules."""

from pinenut.core.layer import (Embedding, LazyLinear, Linear, MLP, Module,
                                Sequential)
from pinenut.core.multigpu import DataParallel
from pinenut.core.tensor import Parameter
from pinenut.nn import functional


class ReLU(Module):
    def forward(self, inputs):
        return functional.relu(inputs)


class LeakyReLU(Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, inputs):
        return functional.leaky_relu(inputs, self.negative_slope)


class Sigmoid(Module):
    def forward(self, inputs):
        return functional.sigmoid(inputs)


class Tanh(Module):
    def forward(self, inputs):
        return functional.tanh(inputs)


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        if not 0 <= p < 1:
            raise ValueError('dropout probability must satisfy 0 <= p < 1')
        self.p = p

    def forward(self, inputs):
        return functional.dropout(inputs, self.p, self.training)


class CrossEntropyLoss(Module):
    def forward(self, inputs, targets):
        return functional.cross_entropy(inputs, targets)


class MSELoss(Module):
    def forward(self, inputs, targets):
        return functional.mse_loss(inputs, targets)


__all__ = [
    'CrossEntropyLoss',
    'DataParallel',
    'Dropout',
    'Embedding',
    'LazyLinear',
    'LeakyReLU',
    'Linear',
    'MLP',
    'MSELoss',
    'Module',
    'Parameter',
    'ReLU',
    'Sequential',
    'Sigmoid',
    'Tanh',
    'functional',
]
