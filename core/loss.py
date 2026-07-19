import pinenut.core as C
import pinenut.core.cuda as cuda


def mse_loss(x0, x1):
    x0, x1 = C.as_tensor(x0), C.as_tensor(x1)
    a = x0 - x1
    y = C.sum(a ** 2) / a.size
    return y


def softmax(x, axis=-1):
    x = C.as_tensor(x)
    xp = cuda.get_array_module(x)
    shifted = x - xp.max(x.data, axis=axis, keepdims=True)
    y = C.exp(shifted)
    y_sum = C.sum(y, axis=axis, keepdims=True)
    return y / y_sum


def cross_entropy(x, t):
    """Return the mean cross-entropy loss for unnormalized logits."""
    x, t = C.as_tensor(x), C.as_tensor(t)
    N = x.shape[0]
    xp = cuda.get_array_module(x)
    shifted = x - xp.max(x.data, axis=1, keepdims=True)
    log_normalizer = C.log(
        C.sum(C.exp(shifted), axis=1, keepdims=True))
    log_p = shifted - log_normalizer
    tlog_p = log_p[xp.arange(N), t.data]
    y = -1 * C.sum(tlog_p) / N
    return y


def binary_cross_entropy(p, t):
    p, t = C.as_tensor(p), C.as_tensor(t)
    if p.ndim != t.ndim:
        t = t.reshape(*p.shape)
    if p.dtype.kind != 'f':
        raise TypeError('probabilities must have a floating-point dtype')
    xp = cuda.get_array_module(p)
    eps = xp.finfo(p.dtype).eps
    p = C.clip(p, eps, 1.0 - eps)
    tlog_p = t * C.log(p) + (1 - t) * C.log(1 - p)
    y = -1 * C.sum(tlog_p) / tlog_p.size
    return y


def binary_cross_entropy_with_logits(x, t):
    x, t = C.as_tensor(x), C.as_tensor(t)
    if x.ndim != t.ndim:
        t = t.reshape(*x.shape)
    abs_x = abs(x)
    positive_x = C.clip(x, 0, float('inf'))
    element_loss = positive_x - x * t + C.log(1 + C.exp(-abs_x))
    y = C.sum(element_loss) / element_loss.size
    return y


def accuracy(y, t):
    y, t = C.as_tensor(y), C.as_tensor(t)
    y_pred = y.data.argmax(axis=1).reshape(t.shape)
    compare = (y_pred == t.data)
    acc = compare.mean()
    acc = C.as_tensor(acc)
    return acc
