import numpy as np
from matplotlib import pyplot as plt
from pinenut import (SpiralDataset, DataLoader, build_graph, relu, accuracy,
                     MLP, softmax, softmax_cross_entropy, Adam, SGD)

dataset = SpiralDataset(300, 3, xdatatype=np.float64, ydatatype=np.int32)
dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

model = MLP((50, 50, 3), hidden_activation=relu, output_activation=softmax)
model.summary()
optimizer = Adam(model)
model.train(dataset, epochs=50, batch_size=20, optimizer=optimizer)
model.save_weights('spiral_mlp.npz')


def predict(x):
    y = model(x)
    return np.argmax(y.data, axis=1)


def plot_decision_boundary(X):
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)

    for i in range(len(X)):
        if dataset.label[i] == 0:
            c='r'
        elif dataset.label[i] == 1:
            c = 'g'
        else:
            c = 'b'
        plt.scatter(X[i, 0], X[i, 1], c=c)
    plt.show()


plot_decision_boundary(dataset.data)
