import numpy as np
import pinenut.core as C


def mean_squared_error(x0, x1):
    a = x0 - x1
    y = C.sum(a ** 2) / len(a)
    return y


def softmax(x, axis=-1):
    x = C.as_tensor(x)
    y = C.exp(x)
    y_sum = C.sum(y, axis=axis, keepdims=True)
    return y / y_sum


def softmax_cross_entropy(x, t):
    x, t = C.as_tensor(x), C.as_tensor(t)
    N = x.shape[0]
    p = softmax(x)
    p = C.clip(p, 1e-15, 1.0)
    log_p = C.log(p)
    tlog_p = log_p[np.arange(N), t.data]
    y = -1 * C.sum(tlog_p) / N
    return y


def binary_cross_entropy(p, t):
    if p.ndim != t.ndim:
        t = t.reshape(*p.shape)
    p = C.clip(p, 1e-15, 1.0)
    tlog_p = t * C.log(p) + (1 - t) * C.log(1 - p)
    y = -1 * C.sum(tlog_p) / len(p)
    return y


def sigmoid_cross_entropy(x, t):
    if x.ndim != t.ndim:
        t = t.reshape(*x.shape)
    x, t = C.as_tensor(x), C.as_tensor(t)
    p = C.sigmoid(x)
    y = binary_cross_entropy(p, t)
    return y


def accuracy(y, t):
    y, t = C.as_tensor(y), C.as_tensor(t)
    y_pred = y.data.argmax(axis=1).reshape(t.shape)
    compare = (y_pred == t.data)
    acc = compare.mean()
    acc = C.as_tensor(acc)
    return acc