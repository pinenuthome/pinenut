import numpy as np
from pinenut import (relu, softmax, ParallelModel)
import pinenut.core.cuda as cuda
import pinenut.core.datasets as dss
import time


def data_transform(x):
    x = x.flatten()
    x = x.astype(np.float32)
    return x / 255.0


epochs = 5
batch_size = 100
cuda_is_available = cuda.Cuda.available()
print(cuda_is_available)
device_list = [0, 1]

train = dss.MNIST(train=True, data_transform=data_transform)
start_time = time.time()
model = ParallelModel(device_list, layer_output_sizes=(784, 100, 10), hidden_activation=relu, output_activation=softmax)
model.train_gpus(train, epochs=5, batch_size=100)
end_time = time.time()
run_time = end_time - start_time
print("elapsed time:", run_time)