from pinenut import Linear, SGD
from pinenut.core import Tensor, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import pinenut.core.cuda as cuda

# generate a linear dataset
np.random.seed(0)
x = np.random.randn(100, 1)
y = 6 + 3 * x + np.random.randn(100, 1)  # noise added

if cuda.Cuda.available():
    x, y = cuda.Cuda.as_cupy(x), cuda.Cuda.as_cupy(y)

x, y = Tensor(x), Tensor(y)  # convert to Tensor

linear = Linear(1, 1)
optimizer = SGD(linear, lr=0.1)

if cuda.Cuda.available():
    linear.to_gpu() 

loss_data = []
for i in range(10):
    y_pred = linear(x)  # forward
    loss = mean_squared_error(y_pred, y)
    loss.backward()
    optimizer.update() # update parameters
    linear.clear_all_grad() # clear gradients
    loss_data.append(loss.data)

# plot the result
x.data = cuda.Cuda.as_numpy(x.data)
y.data = cuda.Cuda.as_numpy(y.data)
linear.to_cpu()

plt.title('Linear Regression')
plt.scatter(x.data, y.data, c='b', marker='o')
xx = np.arange(-3, 3, 0.01)
yy = linear(Tensor(xx[:, np.newaxis])).data
plt.plot(xx, yy, c='r')
plt.show()

# plot the loss
loss_data2 = []
for i in loss_data:
    loss_data2.append(cuda.Cuda.as_numpy(i))

plt.title('Loss')
plt.plot(loss_data2)
plt.show()
