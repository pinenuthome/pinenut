import numpy as np
import matplotlib.pyplot as plt
import pinenut as pn
from pinenut.nn import functional as F

# generate a linear dataset
np.random.seed(0)
x = np.random.randn(100, 1)
y = 6 + 3 * x + np.random.randn(100, 1)  # noise added
x, y = pn.Tensor(x), pn.Tensor(y)  # convert to Tensor

# initialize parameters
W = pn.Tensor(np.random.randn(1, 1))
b = pn.Tensor(np.random.randn(1))


# define a linear function
def predict(x):
    return pn.matmul(x, W) + b


lr = 0.1

loss_data = []
for i in range(10):
    y_pred = predict(x)  # forward
    loss = F.mse_loss(y_pred, y)
    loss.backward()

    # update parameters
    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data

    # clear gradients
    W.zero_grad()
    b.zero_grad()

    loss_data.append(loss.data)


# plot the result
plt.title('Linear Regression')
plt.scatter(x.data, y.data, c='b', marker='o')
xx = np.arange(-3, 3, 0.01)
yy = xx * W.data[0][0] + b.data[0]
plt.plot(xx, yy, c='r')
plt.show()

# plot the loss
plt.title('Loss')
plt.plot(loss_data)
plt.show()
