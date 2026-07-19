import time

import numpy as np

import pinenut as pn
from pinenut import datasets, nn, optim
from pinenut.nn import functional as F


def data_transform(value):
    return value.flatten().astype(np.float32) / 255.0


train_dataset = datasets.MNIST(train=True, data_transform=data_transform)
test_dataset = datasets.MNIST(train=False, data_transform=data_transform)

epochs = 5
batch_size = 100
device = 'cuda' if pn.cuda.is_available() else 'cpu'

model = nn.MLP(784, (100, 10), hidden_activation=F.relu)
optimizer = optim.SGD(model.parameters(), lr=0.1)

start = time.time()
model.fit(
    train_dataset,
    epochs,
    batch_size,
    optimizer,
    test_dataset=test_dataset,
    device=device,
)
print('elapsed time:', time.time() - start)
pn.save(model.state_dict(), 'mnist_weights.npz')
