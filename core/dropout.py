from pinenut.core import Operator
import pinenut.core.cuda as cuda


class Dropout(Operator):
    """
    This class is used to compute the Dropout of a tensor.
    """

    def __init__(self, p=0.5):
        super().__init__()
        p = float(p)
        if not 0 <= p < 1:
            raise ValueError('dropout probability must satisfy 0 <= p < 1')
        self.mask = None
        self.p = p
        self.training = None

    def forward(self, x, training=True):
        self.training = training
        if self.training:
            xp = cuda.get_array_module(x)
            keep_probability = 1 - self.p
            self.mask = xp.random.binomial(1, keep_probability, size=x.shape)
            return x * self.mask / keep_probability
        return x

    def backward(self, grad_output):
        if not self.training:
            return grad_output
        return grad_output * self.mask / (1 - self.p)


def dropout(x, p=0.5, training=True):
    return Dropout(p)(x, training)
