"""Synchronous data-parallel training for an MLP."""

import numpy as np

from pinenut.core import cuda
from pinenut.core.dataset import DataLoader
from pinenut.core.layer import MLP
from pinenut.core.loss import accuracy, cross_entropy
from pinenut.core.optimizer import Adam


class DataParallel:
    def __init__(self, device_ids, layer_sizes=(), hidden_activation=None,
                 output_activation=None):
        if len(device_ids) <= 1:
            raise ValueError('DataParallel requires at least two GPUs')
        if len(set(device_ids)) != len(device_ids):
            raise ValueError('device_ids must not contain duplicates')

        self.device_ids = list(device_ids)
        self.layer_sizes = tuple(layer_sizes)
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.replicas = {}
        self.module = None

    @staticmethod
    def _parameter_dict(module):
        return dict(module.named_parameters())

    def _copy_parameters(self, source, target, target_device, cp):
        source_parameters = self._parameter_dict(source)
        target_parameters = self._parameter_dict(target)
        if source_parameters.keys() != target_parameters.keys():
            raise ValueError('model replicas do not have matching parameters')

        with cp.cuda.Device(target_device):
            for name, source_parameter in source_parameters.items():
                target_parameters[name].data = cp.asarray(source_parameter.data)

    def _average_gradients(self, replicas, primary_device, cp):
        primary_parameters = self._parameter_dict(replicas[primary_device])
        with cp.cuda.Device(primary_device):
            for device in self.device_ids[1:]:
                replica_parameters = self._parameter_dict(replicas[device])
                for name, primary_parameter in primary_parameters.items():
                    primary_parameter.grad.data += cp.asarray(
                        replica_parameters[name].grad.data)

            replica_count = len(self.device_ids)
            for parameter in primary_parameters.values():
                parameter.grad.data /= replica_count

    def fit(self, train_dataset, epochs, batch_size):
        if not cuda.is_available():
            raise RuntimeError('CuPy is required for multi-GPU training')
        device_count = len(self.device_ids)
        if batch_size < device_count or batch_size % device_count != 0:
            raise ValueError('batch_size must be divisible by the number of GPUs')

        sample, _ = train_dataset[0]
        sample_shape = np.asarray(sample).shape
        if not sample_shape:
            raise ValueError('training examples must have at least one dimension')
        in_features = sample_shape[-1]

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        if len(train_loader) == 0:
            raise ValueError('dataset must contain at least one complete batch')

        cp = cuda.cupy()
        primary_device = self.device_ids[0]
        replicas = {}
        for device in self.device_ids:
            with cp.cuda.Device(device):
                model = MLP(
                    in_features,
                    self.layer_sizes,
                    hidden_activation=self.hidden_activation,
                    output_activation=self.output_activation,
                )
                model.cuda(device)
                replicas[device] = model

        for device in self.device_ids[1:]:
            self._copy_parameters(
                replicas[primary_device], replicas[device], device, cp)

        self.replicas = replicas
        self.module = replicas[primary_device]
        optimizer = Adam(self.module.parameters())
        train_loader.cuda(device_list=self.device_ids)

        for epoch in range(epochs):
            total_loss = 0.0
            total_accuracy = 0.0
            seen_samples = 0

            for inputs, labels in train_loader:
                for replica in replicas.values():
                    replica.zero_grad()

                predictions = {}
                losses = {}
                for index, device in enumerate(self.device_ids):
                    with cp.cuda.Device(device):
                        predictions[device] = replicas[device](inputs[index])
                        losses[device] = cross_entropy(
                            predictions[device], labels[index])

                for device in self.device_ids:
                    with cp.cuda.Device(device):
                        losses[device].backward()

                self._average_gradients(replicas, primary_device, cp)
                with cp.cuda.Device(primary_device):
                    optimizer.step()

                for device in self.device_ids[1:]:
                    self._copy_parameters(
                        replicas[primary_device], replicas[device], device, cp)

                for index, device in enumerate(self.device_ids):
                    with cp.cuda.Device(device):
                        batch_count = len(labels[index])
                        batch_accuracy = accuracy(
                            predictions[device], labels[index])
                        total_loss += float(cuda.as_numpy(
                            losses[device].data)) * batch_count
                        total_accuracy += float(cuda.as_numpy(
                            batch_accuracy.data)) * batch_count
                        seen_samples += batch_count
                        losses[device].unchain_backward()

            print('epoch:{}'.format(epoch + 1))
            print('train loss: {:.4f}, accuracy:{:.4f}'.format(
                total_loss / seen_samples,
                total_accuracy / seen_samples))

        return self.module
