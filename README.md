# pinenut
A deep learning framework, which was developed by Neil Jiang.
It supports define-by-run computational graph and automatically calculate the gradients of functions.
It supports GPU acceleration.
It is powerful, flexible and easy to understand.

## Installation
pip install pinenut

## Usage
```python
from pinenut import Tensor
import numpy as np

# define a computational graph
x = Tensor(np.array([1, 2, 3]))
y = Tensor(np.array([4, 5, 6]))
z = x + y
z.backward() # calculate the gradients of z with respect to x and y
print(x.grad) # [1, 1, 1]
print(y.grad) # [1, 1, 1]
```

## GPU acceleration
```python
from pinenut import Tensor, Cuda, matmul, as_array

x = Tensor([1, 2, 3])
y = Tensor([4, 5, 6])

cuda_is_available = Cuda.available()
if cuda_is_available:
    x.to_gpu()
    y.to_gpu()
z = matmul(x, y.T)
assert z.data == as_array(32)
print(type(z.data))
z.backward()
assert (x.grad.data == [4, 5, 6]).all()
assert (y.grad.data == [1, 2, 3]).all()
```

## examples
- [mnist]
```python
import numpy as np
import pinenut.core.datasets as dss
from pinenut import MLP, SGD, relu, softmax
from pinenut import Cuda

def data_transform(x):
    x = x.flatten()
    x = x.astype(np.float32)
    return x / 255.0

train = dss.MNIST(train=True, data_transform=data_transform)
test = dss.MNIST(train=False, data_transform=data_transform)

epochs = 5
batch_size = 100
lr = 0.1 # learning rate

model = MLP([784, 100, 10], hidden_activation=relu, output_activation=softmax)
optimizer = SGD(model, lr)
cuda_is_available = Cuda.available()
model.train(train, epochs, batch_size, optimizer, test, enable_cuda=cuda_is_available)
model.save_weights('mnist_weights.npz')
```

## Features
- [1] Define-by-run computational graph
- [2] GPU acceleration
- [3] Automatic gradient calculation
- [4] Support for various activation, loss and optimizer functions
- [5] Pure Python implementation