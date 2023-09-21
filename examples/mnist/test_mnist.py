import numpy as np
import pinenut.core.datasets as dss
import pinenut.core.dataset as ds
import pinenut.core as C
from pinenut import MLP, SGD, relu, softmax, softmax_cross_entropy, accuracy


def data_transform(x):
    x = x.flatten()
    x = x.astype(np.float32)
    return x / 255.0


test = dss.MNIST(train=False, data_transform=data_transform)

model = MLP([784, 100, 10], hidden_activation=relu, output_activation=softmax)
model.init_model_weight(in_features=784)
model.load_weights('mnist_weights.npz')

batch_size = 100

if test is None:
    print('test dataset is None')
    exit()

with C.no_grad():
    print('---------------------------------------------------------------')
    sum_loss = 0.0
    sum_acc = 0.0

    test_loader = ds.DataLoader(test, batch_size=batch_size, shuffle=True)
    for x, y in test_loader:
        y_pred = model.predict(x)
        loss = softmax_cross_entropy(y_pred, y)

        acc = accuracy(y_pred, y)
        sum_loss += float(loss.data) * len(y)
        sum_acc += float(acc.data) * len(y)

    print('test loss: {:.4f}, accuracy:{:.4f}'
            .format(sum_loss / len(test), sum_acc / len(test)))
    print('---------------------------------------------------------------')