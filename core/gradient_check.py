
def numerical_gradient(f, x, eps=1e-4):
    x0 = x.data - eps
    x1 = x.data + eps
    y0 = f(x0)
    y1 = f(x1)
    grad = (y1 - y0) / (2 * eps)
    return grad
