from pympler import muppy, summary
import gc
from sys import getrefcount
from pinenut import Tensor, matmul, mean_squared_error, build_graph
import numpy as np


def test_func():
    a = Tensor(np.array([2.0, 3.0]), 'a')
    b = Tensor(np.array([4.0, 5.0]), 'b')
    c = a * b
    c.backward()
    # c.backward(enable_buildgraph=False)
    c.unchain_backward()
    print('getrefcount(c)=', getrefcount(c))

def test_train():
    # generate a linear dataset
    np.random.seed(0)
    x = np.random.randn(100, 1)
    y = 6 + 3 * x + np.random.randn(100, 1)  # noise added
    x, y = Tensor(x, 'x'), Tensor(y, 'y')  # convert to Tensor

    # initialize parameters
    W = Tensor(np.random.randn(1, 1), 'W')
    b = Tensor(np.random.randn(1), 'b')

    # define a linear function
    def predict(x):
        return matmul(x, W) + b

    lr = 0.1

    loss_data = []
    for i in range(2):
        y_pred = predict(x)  # forward
        y_pred.name = 'y_pred'
        loss = mean_squared_error(y_pred, y)
        build_graph(loss, 'test_if_memory_leak.png', view=False)
        loss.name = 'loss'
        loss.backward(enable_buildgraph=False)
        loss.unchain_backward()

        # update parameters
        W.data -= lr * W.grad.data
        b.data -= lr * b.grad.data

        # clear gradients
        W.clear_grad()
        b.clear_grad()

        loss_data.append(loss.data)

    # --------------------------
    objs = muppy.get_objects()
    summ = summary.summarize(objs)
    summary.print_(summ)

    print('----  objects of Tensor in the memory ----')
    dicts = [ao for ao in objs if isinstance(ao, Tensor)]
    print('len=', len(dicts))
    for d in dicts:
        print(d.name, d.label, d.creator, id(d))
    # ---------------------------


if __name__ == "__main__":
    test_func()
    test_train()

    gc.collect()

    objs = muppy.get_objects()
    summ = summary.summarize(objs)
    summary.print_(summ)

    print('---- after gc.collect() object of Tensor in memory ----')
    dicts = [ao for ao in objs if isinstance(ao, Tensor)]
    print('len=', len(dicts))
    # np_dicts = [ao for ao in objs if isinstance(ao, np.ndarray)]
    for d in dicts:
        print(d.name, d.label, d.data, d.creator, id(d))
