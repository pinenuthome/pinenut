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
enable_cuda = Cuda.available()
model.train(train, epochs, batch_size, optimizer, test, enable_cuda=enable_cuda)
model.save_weights('mnist_weights.npz')
