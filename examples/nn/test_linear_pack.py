import matplotlib.pyplot as plt
import numpy as np

import pinenut as pn
from pinenut import nn, optim
from pinenut.nn import functional as F


np.random.seed(0)
x = pn.Tensor(np.random.randn(100, 1))
y = pn.Tensor(6 + 3 * x.data + np.random.randn(100, 1))

device = 'cuda' if pn.cuda.is_available() else 'cpu'
x.to(device)
y.to(device)

linear = nn.Linear(1, 1).to(device)
optimizer = optim.SGD(linear.parameters(), lr=0.1)

loss_history = []
for _ in range(10):
    optimizer.zero_grad()
    prediction = linear(x)
    loss = F.mse_loss(prediction, y)
    loss.backward()
    optimizer.step()
    loss_history.append(float(pn.cuda.as_numpy(loss.data)))

x.cpu()
y.cpu()
linear.cpu()

plt.title('Linear Regression')
plt.scatter(x.data, y.data, c='b', marker='o')
xx = np.arange(-3, 3, 0.01)
yy = linear(pn.Tensor(xx[:, np.newaxis])).data
plt.plot(xx, yy, c='r')
plt.show()

plt.title('Loss')
plt.plot(loss_history)
plt.show()
