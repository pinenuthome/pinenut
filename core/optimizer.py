"""Parameter optimizers."""

import collections

from pinenut.core import cuda


class Optimizer:
    def __init__(self, params):
        self.params = []
        parameter_ids = set()
        for parameter in params:
            if id(parameter) not in parameter_ids:
                parameter_ids.add(id(parameter))
                self.params.append(parameter)
        self._hooks = collections.OrderedDict()

    def add_hook(self, hook, name=None):
        hook_name = name or hook.name
        if hook_name in self._hooks:
            raise ValueError('hook {} already exists'.format(hook_name))
        self._hooks[hook_name] = hook

    def remove_hook(self, name):
        if name not in self._hooks:
            raise ValueError('hook {} does not exist'.format(name))
        del self._hooks[name]

    def zero_grad(self):
        for parameter in self.params:
            parameter.zero_grad()

    def step(self):
        parameters = [
            parameter for parameter in self.params
            if parameter.grad is not None
        ]
        if not parameters:
            return

        self._before_step()
        for hook in self._hooks.values():
            hook(parameters)
        for parameter in parameters:
            self._step_parameter(parameter)

    def _before_step(self):
        pass

    def _step_parameter(self, parameter):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self._velocities = {}

    def _step_parameter(self, parameter):
        if self.momentum == 0:
            parameter.data -= self.lr * parameter.grad.data
            return

        key = id(parameter)
        xp = cuda.get_array_module(parameter.data)
        if key not in self._velocities:
            self._velocities[key] = xp.zeros_like(parameter.data)
        velocity = self._velocities[key]
        velocity *= self.momentum
        velocity -= self.lr * parameter.grad.data
        parameter.data += velocity


class Adagrad(Optimizer):
    def __init__(self, params, lr=0.01, eps=1e-8):
        super().__init__(params)
        self.lr = lr
        self.eps = eps
        self._sum_squares = {}

    def _step_parameter(self, parameter):
        key = id(parameter)
        xp = cuda.get_array_module(parameter.data)
        if key not in self._sum_squares:
            self._sum_squares[key] = xp.zeros_like(parameter.data)
        accumulator = self._sum_squares[key]
        accumulator += parameter.grad.data ** 2
        parameter.data -= (
            self.lr * parameter.grad.data
            / (xp.sqrt(accumulator) + self.eps)
        )


class Adam(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.step_count = 0
        self._first_moments = {}
        self._second_moments = {}

    def _before_step(self):
        self.step_count += 1

    def _step_parameter(self, parameter):
        key = id(parameter)
        xp = cuda.get_array_module(parameter.data)
        if key not in self._first_moments:
            self._first_moments[key] = xp.zeros_like(parameter.data)
            self._second_moments[key] = xp.zeros_like(parameter.data)

        first = self._first_moments[key]
        second = self._second_moments[key]
        first *= self.beta1
        first += (1 - self.beta1) * parameter.grad.data
        second *= self.beta2
        second += (1 - self.beta2) * parameter.grad.data ** 2

        first_hat = first / (1 - self.beta1 ** self.step_count)
        second_hat = second / (1 - self.beta2 ** self.step_count)
        parameter.data -= (
            self.lr * first_hat / (xp.sqrt(second_hat) + self.eps)
        )
