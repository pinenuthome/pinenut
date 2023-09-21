import collections
import numpy as np


class Optimizer:
    def __init__(self, target=None):
        self.target = target
        self._hooks = collections.OrderedDict()

    def setup(self, target):
        self.target = target

    def add_hook(self, hook, name=None):
        if name is None:
            name = hook.name
        if name in self._hooks:
            raise ValueError('hook %s already exists' % name)
        self._hooks[name] = hook

    def remove_hook(self, name):
        if name not in self._hooks:
            raise ValueError('hook %s does not exist' % name)
        del self._hooks[name]

    def update(self):
        params = [p for p in self.target.params if p.grad is not None]

        for fn in self._hooks.values():
            fn(params)

        for p in params:
            self.update_one(p)

    def update_one(self, p):
        raise NotImplementedError()


class SGD(Optimizer):
    def __init__(self, target=None, lr=0.01):
        super().__init__(target)
        self.lr = lr

    def update_one(self, p):
        p.data -= self.lr * p.grad.data


class AdaGrad(Optimizer):
    def __init__(self, target=None, lr=0.01, eps=1e-8):
        super().__init__(target)
        self.lr = lr
        self.eps = eps
        self.ms = {}

    def update_one(self, p):
        m_key = id(p)

        if m_key not in self.ms:
            self.ms[m_key] = np.zeros_like(p.data)

        self.ms[m_key] += p.grad.data ** 2
        p.data -= self.lr * p.grad.data / (np.sqrt(self.ms[m_key]) + self.eps)


class Adam(Optimizer):
    def __init__(self, target=None, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(target)
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.ms = {}
        self.vs = {}

    def update_one(self, p):
        self.t += 1
        m_key = id(p)

        if m_key not in self.ms:
            self.ms[m_key] = np.zeros_like(p.data)
            self.vs[m_key] = np.zeros_like(p.data)

        self.ms[m_key] = self.beta1 * self.ms[m_key] + (1 - self.beta1) * p.grad.data
        self.vs[m_key] = self.beta2 * self.vs[m_key] + (1 - self.beta2) * p.grad.data ** 2

        m_hat = self.ms[m_key] / (1 - self.beta1 ** self.t)
        v_hat = self.vs[m_key] / (1 - self.beta2 ** self.t)

        p.data -= self.alpha * m_hat / (np.sqrt(v_hat) + self.eps)


class Momentum(Optimizer):
    def __init__(self, target=None, lr=0.01, momentum=0.9):
        super().__init__(target)
        self.lr = lr
        self.momentum = momentum
        self.vs = {}

    def update_one(self, p):
        v_key = id(p)
        if v_key not in self.vs:
            self.vs[v_key] = np.zeros_like(p.data)

        self.vs[v_key] = self.momentum * self.vs[v_key] - self.lr * p.grad.data
        p.data += self.vs[v_key]