# Pinenut

[![PyPI](https://img.shields.io/pypi/v/pinenut.svg)](https://pypi.org/project/pinenut/)
[![Python](https://img.shields.io/pypi/pyversions/pinenut.svg)](https://pypi.org/project/pinenut/)
[![Tests](https://github.com/pinenuthome/pinenut/actions/workflows/tests.yml/badge.svg)](https://github.com/pinenuthome/pinenut/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/pinenuthome/pinenut/blob/main/LICENSE)

English | [简体中文](https://github.com/pinenuthome/pinenut/blob/main/README.zh-CN.md)

Pinenut is a compact, define-by-run deep learning framework written in Python.
It provides automatic differentiation, neural-network modules, optimization
algorithms, and a shared NumPy/CuPy API for CPU and NVIDIA GPU execution. Its
small codebase is designed to be easy to read, experiment with, and extend.

## Highlights

- Dynamic computational graphs with first- and higher-order gradients
- A consistent `Module`, `Parameter`, and `Optimizer` interface
- Common layers, activations, losses, datasets, and optimizers
- NumPy execution on CPU and optional CuPy acceleration on NVIDIA GPUs
- Model state dictionaries and synchronous multi-GPU training
- Pure Python implementation suitable for learning and small experiments

## Requirements

- Python 3.8 or later
- NumPy 1.20 or later
- Optional GPU support: a compatible NVIDIA driver and exactly one supported
  CuPy package

## Installation

Install the latest release from PyPI:

```bash
python -m pip install --upgrade pinenut
```

Or install the current source tree in editable mode:

```bash
git clone https://github.com/pinenuthome/pinenut.git
cd pinenut
python -m pip install -e .
```

For NVIDIA GPUs, install the extra matching your CUDA major version:

```bash
# CUDA 12.x
python -m pip install "pinenut[gpu-cu12]"

# CUDA 13.x
python -m pip install "pinenut[gpu-cu13]"
```

Install only one CuPy distribution in an environment. For virtual-environment
setup, distribution-specific prerequisites, CUDA troubleshooting, and a real
two-GPU check, see the
[Linux installation guide](https://github.com/pinenuthome/pinenut/blob/main/docs/linux-installation.md).

## Automatic differentiation

Build a dynamic graph and differentiate a scalar result:

```python
import numpy as np

import pinenut as pn

x = pn.tensor(np.array([1.0, 2.0, 3.0]))
loss = pn.sum(x ** 2)
loss.backward()

print(loss.data)    # 14.0
print(x.grad.data)  # [2. 4. 6.]
```

Set `create_graph=True` when a later computation needs to differentiate the
resulting gradient:

```python
x = pn.tensor(3.0)
y = x ** 2
y.backward(create_graph=True)

first_derivative = x.grad
x.zero_grad()
first_derivative.backward()
print(x.grad.data)  # 2.0
```

## Neural-network training

Modules expose parameters to optimizers and use explicit training and
evaluation modes:

```python
import numpy as np

import pinenut as pn
from pinenut import nn, optim
from pinenut.nn import functional as F

model = nn.Sequential(
    nn.Linear(2, 32),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(32, 3),
)
optimizer = optim.Adam(model.parameters(), lr=0.001)

inputs = pn.tensor(np.random.randn(8, 2))
targets = pn.tensor(np.random.randint(0, 3, size=8))

model.train()
optimizer.zero_grad()
logits = model(inputs)
loss = F.cross_entropy(logits, targets)
loss.backward()
optimizer.step()

model.eval()
with pn.no_grad():
    predictions = model(inputs)
```

Available components include `Linear`, `LazyLinear`, `Sequential`, `MLP`,
`Embedding`, `Dropout`, `SGD`, `Adagrad`, and `Adam`. `SGD` accepts an optional
`momentum` value. Classification losses accept unnormalized logits, so the
final layer should not apply softmax when used with `F.cross_entropy`.

## Devices

Use the same model and tensor API on CPU and GPU:

```python
device = 'cuda' if pn.cuda.is_available() else 'cpu'
model.to(device)
inputs.to(device)
targets.to(device)
```

Specific GPUs can be selected with names such as `cuda:1`. Calling `cpu()`,
`cuda()`, or `to(device)` moves a tensor or module in place and returns it.

## Saving model state

```python
pn.save(model.state_dict(), 'model_weights.npz')

restored = nn.Sequential(
    nn.Linear(2, 32),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(32, 3),
)
restored.load_state_dict(pn.load('model_weights.npz'))
restored.eval()
```

## Examples

- [MNIST training](https://github.com/pinenuthome/pinenut/blob/main/examples/mnist/train_mnist.py)
- [MNIST multi-GPU training](https://github.com/pinenuthome/pinenut/blob/main/examples/mnist/train_mnist_gpus.py)
- [Spiral classification](https://github.com/pinenuthome/pinenut/blob/main/examples/nn/test_spiral.py)
- [Embedding example](https://github.com/pinenuthome/pinenut/blob/main/examples/nn/test_embedding.py)
- [Dropout example](https://github.com/pinenuthome/pinenut/blob/main/examples/nn/test_dropout.py)

Run the Spiral visualization to train an MLP and save its decision boundary,
metrics, and state dictionary:

```bash
python examples/nn/test_spiral.py
python examples/nn/test_spiral.py --animate
```

Outputs are written to `demo_outputs/spiral/`. Use `--help` to configure the
dataset, model, training schedule, output directory, and CPU/GPU selection.

## Tests

Install the development dependencies and run the CPU-compatible test suite:

```bash
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-deps
python -m unittest discover -s test -v
```

On a machine with two physical NVIDIA GPUs, run the synthetic multi-GPU smoke
test:

```bash
CUDA_VISIBLE_DEVICES=0,1 python tools/test_multigpu.py --devices 0,1
```

To test physical GPUs 2 and 3, remap them to logical devices 0 and 1:

```bash
CUDA_VISIBLE_DEVICES=2,3 python tools/test_multigpu.py --devices 0,1
```

The command prints `MULTI-GPU SMOKE TEST PASSED` when device placement,
training, parameter synchronization, and finite-value checks all succeed.

## Project structure

```text
core/       Automatic differentiation and internal implementations
nn/         Neural-network modules and stateless functions
optim/      Optimization algorithms
datasets/   Built-in datasets
utils/      Data loading and utility interfaces
examples/   Runnable training and visualization examples
test/       Unit and regression tests
tools/      Standalone verification utilities
docs/       Installation and usage guides
```

## Contributing

Issues and pull requests are welcome. Please add or update tests for behavioral
changes and run the test suite before submitting a pull request.

## License

Pinenut is released under the
[MIT License](https://github.com/pinenuthome/pinenut/blob/main/LICENSE).
