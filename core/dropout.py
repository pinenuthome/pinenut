import numpy as np

from pinenut.core import Operator
import pinenut.core.utils as U


class Dropout(Operator):
    """
    This class is used to compute the Dropout of a tensor.
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.mask = None
        self.p = p

    def forward(self, x):
        if U.Config.train:
            self.mask = np.random.binomial(1, (1 - self.p), size=x.shape)
            y = x * self.mask
        else:
            y = x
        return y

    def backward(self, grad_output):
        grad_x = grad_output * self.mask
        return grad_x


def dropout(x, p=0.5):
    return Dropout(p)(x)