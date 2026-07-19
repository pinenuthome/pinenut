import argparse

import numpy as np

import pinenut as pn
from pinenut import nn
from pinenut.nn import functional as F


class SyntheticDataset:
    def __init__(self, sample_count, feature_count=4, seed=7):
        rng = np.random.default_rng(seed)
        self.data = rng.normal(size=(sample_count, feature_count)).astype(np.float32)
        decision = self.data[:, 0] - 0.5 * self.data[:, 1] + self.data[:, 2]
        self.labels = (decision > 0).astype(np.int32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


def parse_devices(value):
    try:
        devices = [int(device.strip()) for device in value.split(',')]
    except ValueError as error:
        raise argparse.ArgumentTypeError('devices must be comma-separated integers') from error

    if len(devices) != 2 or len(set(devices)) != 2:
        raise argparse.ArgumentTypeError('exactly two distinct GPU IDs are required')
    return devices


def verify_replicas(models, devices):
    master_params = {
        name: pn.cuda.as_numpy(param.data)
        for name, param in models[devices[0]].named_parameters()
    }

    for device in devices:
        for name, param in models[device].named_parameters():
            if int(param.data.device.id) != device:
                raise AssertionError(
                    '{} is on GPU {}, expected GPU {}'.format(
                        name, param.data.device.id, device))

            value = pn.cuda.as_numpy(param.data)
            if not np.isfinite(value).all():
                raise AssertionError('{} contains non-finite values'.format(name))
            np.testing.assert_allclose(
                value, master_params[name], rtol=1e-6, atol=1e-7,
                err_msg='replica mismatch for {} on GPU {}'.format(name, device))


def main():
    parser = argparse.ArgumentParser(description='Run a real two-GPU pinenut smoke test')
    parser.add_argument('--devices', type=parse_devices, default=parse_devices('0,1'))
    parser.add_argument('--samples', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=3)
    args = parser.parse_args()

    if not pn.cuda.is_available():
        raise RuntimeError('CuPy is not installed; follow docs/linux-installation.md')

    cp = pn.cuda.cupy()
    visible_count = cp.cuda.runtime.getDeviceCount()
    if max(args.devices) >= visible_count:
        raise RuntimeError(
            'requested GPUs {}, but only {} GPU(s) are visible'.format(
                args.devices, visible_count))

    if args.epochs < 1:
        raise ValueError('epochs must be at least one')
    if args.batch_size < len(args.devices):
        raise ValueError('batch-size must be at least the GPU count')
    if args.samples < args.batch_size:
        raise ValueError('samples must be at least batch-size')
    if args.batch_size % len(args.devices) != 0:
        raise ValueError('batch-size must be divisible by the GPU count')

    print('visible GPUs: {}'.format(visible_count))
    for device in args.devices:
        properties = cp.cuda.runtime.getDeviceProperties(device)
        name = properties['name']
        if isinstance(name, bytes):
            name = name.decode()
        print('using logical GPU {}: {}'.format(device, name))

    dataset = SyntheticDataset(args.samples)
    parallel_model = nn.DataParallel(
        args.devices, layer_sizes=(16, 2), hidden_activation=F.relu)
    parallel_model.fit(
        dataset, epochs=args.epochs, batch_size=args.batch_size)
    verify_replicas(parallel_model.replicas, args.devices)

    print('MULTI-GPU SMOKE TEST PASSED')


if __name__ == '__main__':
    main()
