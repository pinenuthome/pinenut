import time

import numpy as np

from pinenut import datasets, nn
from pinenut.nn import functional as F


def data_transform(value):
    return value.flatten().astype(np.float32) / 255.0


train_dataset = datasets.MNIST(train=True, data_transform=data_transform)
model = nn.DataParallel(
    [0, 1],
    layer_sizes=(100, 10),
    hidden_activation=F.relu,
)

start = time.time()
model.fit(train_dataset, epochs=5, batch_size=100)
print('elapsed time:', time.time() - start)
