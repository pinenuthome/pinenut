from pinenut.core import Operator
import numpy as np


class Sin(Operator):
    """
    This class is used to compute the sine of a tensor.
    """
    @property
    def label(self):
        return '__sin__'

    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, grad_output):
        x = self.inputs[0]
        grad_x = grad_output * cos(x)
        return grad_x


def sin(x):
    return Sin()(x)


class Cos(Operator):
    """
    This class is used to compute the cosine of a tensor.
    """
    @property
    def label(self):
        return '__cos__'

    def forward(self, x):
        y = np.cos(x)
        return y

    def backward(self, grad_output):
        x = self.inputs[0]
        grad_x = -grad_output * sin(x)
        return grad_x


def cos(x):
    return Cos()(x)


class Tanh(Operator):
    """
    This class is used to compute the hyperbolic tangent of a tensor.
    """
    @property
    def label(self):
        return '__tanh__'

    def forward(self, x):
        y = np.tanh(x)
        return y

    def backward(self, grad_output):
        y = self.outputs[0]()
        grad_x = grad_output * (1 - y * y)
        return grad_x


def tanh(x):
    return Tanh()(x)
