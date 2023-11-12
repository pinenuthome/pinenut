from pinenut.core import Operator
import pinenut.core.cuda as cuda


class Sigmoid(Operator):
    """
    This class is used to compute the sigmoid of a tensor.
    """

    def forward(self, x):
        xp = cuda.Cuda.get_array_module(x)
        y = 1 / (1 + xp.exp(-x))
        return y

    def backward(self, grad_output):
        y = self.outputs[0]()
        grad_x = grad_output * y * (1 - y)
        return grad_x


def sigmoid(x):
    return Sigmoid()(x)


class ReLu(Operator):
    """
    This class is used to compute the ReLu of a tensor.
    """

    def forward(self, x):
        xp = cuda.Cuda.get_array_module(x)
        y = xp.maximum(0, x)
        return y

    def backward(self, grad_output):
        x, = self.inputs
        grad_x = grad_output * (x > 0)
        return grad_x


def relu(x):
    return ReLu()(x)


class LeakyRelu(Operator):
    """
    This class is used to compute the Leaky ReLu of a tensor.
    """

    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        xp = cuda.Cuda.get_array_module(x)
        y = xp.maximum(self.alpha * x, x)
        return y

    def backward(self, grad_output):
        x, = self.inputs
        grad_x = grad_output * (x > 0) + grad_output * (x <= 0) * self.alpha
        return grad_x


def leaky_relu(x, alpha=0.01):
    return LeakyRelu(alpha)(x)