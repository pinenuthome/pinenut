"""Neural-network modules and parameter containers."""

import numpy as np

from pinenut.core import cuda
from pinenut.core.dataset import DataLoader
from pinenut.core.loss import accuracy, cross_entropy
from pinenut.core.tensor import Parameter, matmul, no_grad


class Module:
    def __init__(self):
        super().__setattr__('_parameters', set())
        super().__setattr__('_modules', set())
        super().__setattr__('training', True)

    def __setattr__(self, name, value):
        parameters = self.__dict__.get('_parameters')
        modules = self.__dict__.get('_modules')
        if parameters is not None:
            parameters.discard(name)
        if modules is not None:
            modules.discard(name)
        if isinstance(value, Parameter) and parameters is not None:
            parameters.add(name)
        elif isinstance(value, Module) and modules is not None:
            modules.add(name)
        super().__setattr__(name, value)

    def __delattr__(self, name):
        self._parameters.discard(name)
        self._modules.discard(name)
        super().__delattr__(name)

    def __call__(self, *inputs):
        return self.forward(*inputs)

    def forward(self, *inputs):
        raise NotImplementedError

    def parameters(self):
        for _, parameter in self.named_parameters():
            yield parameter

    def named_parameters(self, prefix=''):
        return self._named_parameters(prefix, set(), set())

    def _named_parameters(self, prefix, parameter_ids, module_ids):
        if id(self) in module_ids:
            return
        module_ids.add(id(self))
        for name in sorted(self._parameters):
            parameter = getattr(self, name)
            if id(parameter) in parameter_ids:
                continue
            parameter_ids.add(id(parameter))
            full_name = '{}.{}'.format(prefix, name) if prefix else name
            yield full_name, parameter
        for name in sorted(self._modules):
            child_prefix = '{}.{}'.format(prefix, name) if prefix else name
            module = getattr(self, name)
            yield from module._named_parameters(
                child_prefix, parameter_ids, module_ids)

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()

    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise TypeError('mode must be a boolean')
        self.training = mode
        for name in self._modules:
            getattr(self, name).train(mode)
        return self

    def eval(self):
        return self.train(False)

    def cpu(self):
        for parameter in self.parameters():
            parameter.cpu()
        return self

    def cuda(self, device=None):
        for parameter in self.parameters():
            parameter.cuda(device)
        return self

    def to(self, device):
        if device == 'cpu':
            return self.cpu()
        return self.cuda(device)

    def state_dict(self):
        uninitialized = [
            name for name, parameter in self.named_parameters()
            if parameter.data is None
        ]
        if uninitialized:
            raise ValueError(
                'cannot export uninitialized parameters: {}'.format(
                    ', '.join(uninitialized)))
        return {
            name: np.array(cuda.as_numpy(parameter.data), copy=True)
            for name, parameter in self.named_parameters()
        }

    def load_state_dict(self, state_dict, strict=True):
        named_parameters = dict(self.named_parameters())
        expected = set(named_parameters)
        stored = set(state_dict)
        missing = expected - stored
        unexpected = stored - expected
        if strict and (missing or unexpected):
            raise ValueError(
                'parameter keys do not match: missing={}, unexpected={}'.format(
                    sorted(missing), sorted(unexpected)))

        for name in expected & stored:
            parameter = named_parameters[name]
            value = np.asarray(state_dict[name])
            if parameter.data is not None and parameter.data.shape != value.shape:
                raise ValueError(
                    'parameter shape mismatch for {}: expected {}, got {}'.format(
                        name, parameter.data.shape, value.shape))
            if parameter.device == 'cpu':
                parameter.data = np.asarray(value)
            else:
                parameter.data = cuda.as_cupy(value, device=parameter.device)
        return self


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, dtype=np.float64):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        self.weight = Parameter(None, name='weight')
        self.bias = (
            Parameter(np.zeros(out_features, dtype=dtype), name='bias')
            if bias else None
        )
        self.reset_parameters()

    def reset_parameters(self, xp=None, device=None):
        if xp is None:
            xp = np if self.weight.device == 'cpu' else cuda.cupy()
        if xp is np:
            self._reset_parameters(xp)
            return
        device = self.weight.device if device is None else device
        with xp.cuda.Device(cuda._device_index(device)):
            self._reset_parameters(xp)

    def _reset_parameters(self, xp):
        scale = xp.sqrt(2 / self.in_features)
        self.weight.data = (
            xp.random.randn(self.out_features, self.in_features)
            .astype(self.dtype) * scale
        )
        if self.bias is not None:
            self.bias.data = xp.zeros(self.out_features, dtype=self.dtype)

    def forward(self, inputs):
        output = matmul(inputs, self.weight.T)
        if self.bias is not None:
            output = output + self.bias
        return output


class LazyLinear(Module):
    def __init__(self, out_features, bias=True, dtype=np.float64):
        super().__init__()
        self.in_features = None
        self.out_features = out_features
        self.dtype = dtype
        self.weight = Parameter(None, name='weight')
        self.bias = (
            Parameter(np.zeros(out_features, dtype=dtype), name='bias')
            if bias else None
        )

    def reset_parameters(self, in_features, xp=None, device=None):
        self.in_features = in_features
        if xp is None:
            xp = np if self.weight.device == 'cpu' else cuda.cupy()
        if xp is np:
            self._reset_parameters(xp)
            return
        device = self.weight.device if device is None else device
        with xp.cuda.Device(cuda._device_index(device)):
            self._reset_parameters(xp)

    def _reset_parameters(self, xp):
        scale = xp.sqrt(2 / self.in_features)
        self.weight.data = (
            xp.random.randn(self.out_features, self.in_features)
            .astype(self.dtype) * scale
        )
        if self.bias is not None:
            self.bias.data = xp.zeros(self.out_features, dtype=self.dtype)

    def forward(self, inputs):
        if self.weight.data is None:
            xp = cuda.get_array_module(inputs)
            device = None
            if xp is not np:
                is_tensor = (
                    hasattr(inputs, 'grad') and hasattr(inputs, 'creator'))
                data = inputs.data if is_tensor else inputs
                device = int(data.device.id)
            self.reset_parameters(inputs.shape[-1], xp, device)
        output = matmul(inputs, self.weight.T)
        if self.bias is not None:
            output = output + self.bias
        return output


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self._sequence = []
        for index, module in enumerate(modules):
            if not isinstance(module, Module):
                raise TypeError('Sequential accepts Module instances')
            name = str(index)
            setattr(self, name, module)
            self._sequence.append(name)

    def forward(self, inputs):
        for name in self._sequence:
            inputs = getattr(self, name)(inputs)
        return inputs


class MLP(Module):
    def __init__(self, in_features, layer_sizes, hidden_activation=None,
                 output_activation=None, dtype=np.float64):
        super().__init__()
        self.in_features = in_features
        self.layer_sizes = tuple(layer_sizes)
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layers = []

        current_features = in_features
        for index, out_features in enumerate(self.layer_sizes):
            layer = Linear(current_features, out_features, dtype=dtype)
            name = 'linear{}'.format(index)
            setattr(self, name, layer)
            self.layers.append(name)
            current_features = out_features

    def forward(self, inputs):
        last_index = len(self.layers) - 1
        for index, name in enumerate(self.layers):
            inputs = getattr(self, name)(inputs)
            activation = (
                self.output_activation
                if index == last_index else self.hidden_activation
            )
            if activation is not None:
                inputs = activation(inputs)
        return inputs

    def reset_parameters(self):
        for name in self.layers:
            getattr(self, name).reset_parameters()

    def summary(self):
        print('---------------------------------------------------------------')
        print('Layer (type)             Output Shape              Activation #')
        print('===============================================================')
        last_index = len(self.layers) - 1
        for index, name in enumerate(self.layers):
            layer = getattr(self, name)
            activation = (
                self.output_activation
                if index == last_index else self.hidden_activation
            )
            description = activation.__name__ if activation else 'None'
            print('{:<30} {:<20} {:<25}'.format(
                name, str(layer.out_features), description))
        print('===============================================================')
        print('Total layers: {}'.format(len(self.layers)))
        print('---------------------------------------------------------------')

    def fit(self, train_dataset, epochs, batch_size, optimizer,
            test_dataset=None, device='cpu'):
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        self.to(device)
        train_loader.to(device)

        for epoch in range(epochs):
            self.train()
            total_loss = 0.0
            total_accuracy = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                predictions = self(inputs)
                loss = cross_entropy(predictions, labels)
                loss.backward()
                optimizer.step()

                batch_count = len(labels)
                batch_accuracy = accuracy(predictions, labels)
                total_loss += float(cuda.as_numpy(loss.data)) * batch_count
                total_accuracy += float(cuda.as_numpy(
                    batch_accuracy.data)) * batch_count
                loss.unchain_backward()

            print('epoch:{}'.format(epoch + 1))
            print('train loss: {:.4f}, accuracy:{:.4f}'.format(
                total_loss / len(train_dataset),
                total_accuracy / len(train_dataset)))

            if test_dataset is not None:
                self.eval()
                test_loader = DataLoader(
                    test_dataset, batch_size=batch_size, shuffle=False)
                test_loader.to(device)
                test_loss = 0.0
                test_accuracy = 0.0
                with no_grad():
                    for inputs, labels in test_loader:
                        predictions = self(inputs)
                        loss = cross_entropy(predictions, labels)
                        batch_count = len(labels)
                        batch_accuracy = accuracy(predictions, labels)
                        test_loss += float(cuda.as_numpy(
                            loss.data)) * batch_count
                        test_accuracy += float(cuda.as_numpy(
                            batch_accuracy.data)) * batch_count
                print('test loss: {:.4f}, accuracy:{:.4f}'.format(
                    test_loss / len(test_dataset),
                    test_accuracy / len(test_dataset)))
            print('---------------------------------------------------------------')
        return self

    def predict(self, *inputs):
        with no_grad():
            return self(*inputs)


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, dtype=np.float64):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.dtype = dtype
        self.weight = Parameter(
            np.random.randn(num_embeddings, embedding_dim).astype(dtype)
            * np.sqrt(2 / embedding_dim),
            name='weight',
        )

    def forward(self, inputs):
        return self.weight[inputs]
