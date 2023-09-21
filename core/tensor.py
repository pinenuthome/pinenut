import pinenut.core.backward as backward
import pinenut.core.utils as utils
import weakref
import numpy as np

'''
    Define a tensor class, which is a wrapper of numpy array.
    The tensor class is used to store the data and gradient of a tensor.
    It provides backward function to compute the gradient.
    Tensors and functions are connected by a graph, which has rank.
    override __add__, __sub__, __mul__, __truediv__, __pow__, etc, to support tensor operations easy to use.
'''


def no_grad():
    return utils.using_config('enable_backprop', False)


def no_train():
    return utils.using_config('train', False)


class Tensor:
    _backward_engine = backward.BackwardEngine()

    def __init__(self, data, name=None, creator=None):
        self.data = as_array(data)
        self._name = name
        self.creator = creator
        self.grad = None
        self.rank = 0

    def backward(self, retain_grad=False):
        if self.creator is None:
            return
        if self.grad is None:
            self.grad = as_tensor(np.ones_like(self.data))
        self._backward_engine.run_backward(self, self.grad, retain_grad)

    def set_creator(self, creator):
        self.creator = creator
        self.rank = creator.rank + 1

    def __pos__(self):
        return self

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'Tensor(None)'
        return 'Tensor(%s)' % self.data.__repr__()

    def __str__(self):
        return self.__repr__()

    @property
    def label(self):
        if self.shape == ():
            return str(self.data.dtype)
        return str(self.shape) + ' ' + str(self.data.dtype)

    @property
    def name(self):
        if self._name is None:
            self._name = self.__class__.__name__
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    def __eq__(self, other):
        if isinstance(other, Tensor):
            return np.array_equal(self.data, other.data)
        return np.array_equal(self.data, other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __le__(self, other):
        if isinstance(other, Tensor):
            return np.less_equal(self.data, other.data)
        return np.less_equal(self.data, other)

    def __lt__(self, other):
        if isinstance(other, Tensor):
            return np.less(self.data, other.data)
        return np.less(self.data, other)

    def __ge__(self, other):
        if isinstance(other, Tensor):
            return np.greater_equal(self.data, other.data)
        return np.greater_equal(self.data, other)

    def __gt__(self, other):
        if isinstance(other, Tensor):
            return np.greater(self.data, other.data)
        return np.greater(self.data, other)

    def clear_grad(self):
        self.grad = None

    def unchain(self):
        self.creator = None

    def unchain_backward(self):
        if self.creator is not None:
            ops = [self.creator]
            while ops:
                op = ops.pop()
                for x in op.inputs:
                    if x.creator is not None:
                        ops.append(x.creator)
                        x.unchain()
                op.inputs = []
                op.outputs = []

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def size(self):
        return self.data.size

    @property
    def ndim(self):
        return self.data.ndim

    def reshape(self, shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        return reshape(self, shape)

    def transpose(self, axes=None):
        return transpose(self, axes)

    @property
    def T(self):
        return transpose(self)


class Parameter(Tensor):
    pass


def as_tensor(data, name=None):
    if isinstance(data, Tensor):
        return data
    else:
        return Tensor(data, name=name)


def as_array(data, dtype=None):
    if np.isscalar(data) or isinstance(data, (list, tuple)):
        if dtype is None:
            return np.array(data)
        else:
            return np.array(data, dtype=dtype)
    else:
        if dtype is None:
            return data
        else:
            return data.astype(dtype, copy=False)


class Operator:
    '''
    This class is a base class of all functions, providing a compute graph with operators and tensors.
    It has a forward function to compute the output of the graph and a backward function to compute the gradient.
    '''

    def __init__(self):
        self.inputs = []
        self.outputs = []
        self.rank = 0

    def __call__(self, *inputs):
        return self._do_forward(*inputs)

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def label(self):
        return self.name

    def _do_forward(self, *inputs):
        inputs = [as_tensor(arg) for arg in inputs]
        unpacked_input = [tensor.data for tensor in inputs]
        raw_output = self.forward(*unpacked_input)
        if not isinstance(raw_output, tuple):
            raw_output = (raw_output,)
        outputs = [as_array(a) for a in raw_output]
        outputs = [as_tensor(a) for a in outputs]

        if utils.Config.enable_backprop:
            max_rank = 0
            for x in inputs:
                if x.rank > max_rank:
                    max_rank = x.rank
            self.rank = max_rank

            self.inputs = inputs
            for output in outputs:
                output.set_creator(self)

            self.outputs = [weakref.ref(output) for output in outputs]

        ret = outputs if len(outputs) > 1 else outputs[0]
        return ret

    def forward(self, *input_data):
        raise NotImplementedError

    def backward(self, *output_grad):
        raise NotImplementedError


class Add(Operator):
    '''
    This class is used to add two tensors.
    '''

    def __init__(self):
        super().__init__()
        self.x0_shape = None
        self.x1_shape = None

    @property
    def label(self):
        return '_+_'

    def forward(self, x0, x1):
        self.x0_shape = x0.shape
        self.x1_shape = x1.shape
        y = x0 + x1
        return y

    def backward(self, grad_output):
        grad_x0, grad_x1 = grad_output, grad_output
        if self.x0_shape != self.x1_shape:
            grad_x0 = sum_to(grad_x0, self.x0_shape)
            grad_x1 = sum_to(grad_x1, self.x1_shape)
        return grad_x0, grad_x1


def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)


def radd(x0, x1):
    x0 = as_array(x0)
    return Add()(x1, x0)


class Sub(Operator):
    '''
    This class is used to subtract two tensors.
    '''

    def __init__(self):
        super().__init__()
        self.x1_shape = None
        self.x0_shape = None

    @property
    def label(self):
        return '_-_'

    def forward(self, x1, x2):
        self.x0_shape = x1.shape
        self.x1_shape = x2.shape
        y = x1 - x2
        return y

    def backward(self, grad_output):
        gx0 = grad_output
        gx1 = -grad_output
        if self.x0_shape != self.x1_shape:
            gx0 = sum_to(gx0, self.x0_shape)
            gx1 = sum_to(gx1, self.x1_shape)
        return gx0, gx1


def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)


class Mul(Operator):
    """
    This class is used to multiply two tensors.
    """
    @property
    def label(self):
        return '_*_'

    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, grad_output):
        x0, x1 = self.inputs
        gx0 = grad_output * x1
        gx1 = grad_output * x0
        if x0.shape != x1.shape:
            gx0 = sum_to(gx0, x0.shape)
            gx1 = sum_to(gx1, x1.shape)
        return gx0, gx1


def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)


def rmul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x1, x0)

class Div(Operator):
    """
    This class is used to divide two tensors.
    """
    @property
    def label(self):
        return '_/_'

    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, grad_output):
        x0, x1 = self.inputs
        grad_x0 = grad_output / x1
        grad_x1 = -grad_output * x0 / (x1 * x1)
        if x0.shape != x1.shape:
            grad_x0 = sum_to(grad_x0, x0.shape)
            grad_x1 = sum_to(grad_x1, x1.shape)
        return grad_x0, grad_x1


def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)


class Neg(Operator):
    """
    This class is used to negate a tensor.
    """
    @property
    def label(self):
        return '__neg__'

    def forward(self, x):
        y = -x
        return y

    def backward(self, grad_y):
        return -grad_y


def neg(x):
    return Neg()(x)


class Pow(Operator):
    """
    This class is used to compute the power of a tensor.
    """
    @property
    def label(self):
        return '_**_'

    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        if isinstance(self.n, Tensor):
            n = self.n.data
        else:
            n = self.n
        y = x ** n
        return y

    def backward(self, grad_output):
        x = self.inputs[0]  # 使用 self.inputs[0] 代替 self.inputs，运算符不能识别list类型
        grad_x = self.n * x ** (self.n - 1) * grad_output
        return grad_x


def pow(x, n):
    n = as_tensor(as_array(n))
    return Pow(n)(x)


class Exp(Operator):
    """
    This class is used to compute the exponential of a tensor.
    """
    @property
    def label(self):
        return '__exp__'

    def forward(self, x):
        try:
            y = np.exp(x)
        except:
            print(x)
            raise
        return y

    def backward(self, grad_output):
        y = self.outputs[0]()  # weakref access
        grad_x = y * grad_output
        return grad_x


def exp(x):
    return Exp()(x)


class Abs(Operator):
    """
    This class is used to compute the absolute value of a tensor.
    """
    @property
    def label(self):
        return '|_|'

    def forward(self, x):
        y = np.abs(x)
        return y

    def backward(self, grad_output):
        x = self.inputs[0]
        grad_x = grad_output * (x >= 0) * 2 - grad_output
        return grad_x


def abs(x):
    return Abs()(x)


class Reshape(Operator):
    """
    This class is used to reshape a tensor.
    """
    @property
    def label(self):
        return '__reshape__'

    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        self.x_shape = None

    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, grad_output):
        grad_x = grad_output.reshape(self.x_shape)
        return grad_x


def reshape(x, shape):
    """
    This function is used to reshape a tensor.
    :param x:
    :param shape:
    :return:
    """
    if x.shape == shape:
        return as_tensor(x)
    return Reshape(shape)(x)


class Transpose(Operator):
    """
    This class is used to transpose a tensor.
    """
    @property
    def label(self):
        return '__transpose__'

    def __init__(self, axes=None):
        super().__init__()
        self.axes = axes
        self.x_shape = None

    def forward(self, x):
        self.x_shape = x.shape

        # if x is a vector, expand it to a matrix, Neil 2023-09-13
        is_a_vector = x.ndim == 1
        if is_a_vector:
            y = expand_dims(x, -1)
            return y

        y = x.transpose(self.axes)
        return y

    def backward(self, grad_output):
        inv_axes = self.axes
        if inv_axes:
            axes_len = len(inv_axes)
            inv_axes = tuple(np.argsort([ax % axes_len for ax in inv_axes]))
            return transpose(grad_output, inv_axes)
        else:
            return transpose(grad_output)


def transpose(x, axes=None):
    """
    This function is used to transpose a tensor.
    :param x:
    :param axes:
    :return:
    """
    return Transpose(axes)(x)


class MatMul(Operator):
    """
    This class is used to compute the matrix multiplication of two tensors.
    """
    @property
    def label(self):
        return '_@_'

    def forward(self, a, b):
        y = a.dot(b)
        return y

    def backward(self, grad_output):
        a, b = self.inputs
        grad_x0 = matmul(grad_output, b.T)

        # if grad_output is a vector, expand it to a matrix, Neil 2023-09-13
        if len(grad_output.shape) == 1:
            grad_output = expand_dims(grad_output, -1)
            grad_output = transpose(grad_output)

        aT = a.T
        grad_x1 = matmul(aT, grad_output)
        return grad_x0, grad_x1


def matmul(a, b):
    return MatMul()(a, b)


class ExpandDims(Operator):
    """
    This class is used to expand the dimension of a tensor.
    """

    def __init__(self, axis):
        super().__init__()
        self.axis = int(axis)

    @property
    def label(self):
        return '__expand_dims__'

    def forward(self, x):
        y = np.expand_dims(x, self.axis)
        return y

    def backward(self, grad_output):
        grad_x = reshape(grad_output, self.inputs[0].shape)
        return grad_x


def expand_dims(x, axis):
    return ExpandDims(axis)(x)


class SumTo(Operator):
    """
    This class is used to sum a tensor to a given shape.
    """

    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        self.x_shape = None

    @property
    def label(self):
        return '__sum_to__'

    def forward(self, x):
        self.x_shape = x.shape
        y = util_sum_to(x, self.shape)
        return y

    def backward(self, grad_output):
        grad_x = broadcast_to(grad_output, self.x_shape)
        return grad_x


def sum_to(x, shape):
    '''
    This function is used to sum a tensor to a given shape.
    :param x:
    :param shape:
    :return:
    '''
    if x.shape == shape:
        return as_tensor(x)
    return SumTo(shape)(x)


def util_sum_to(x, shape):
    '''
    This function is used to sum a tensor to a given shape.
    :param x:
    :param shape:
    :return:
    '''
    if x.shape == shape:
        return x
    if isinstance(x, Tensor):
        raise TypeError('x should be a numpy array, not a tensor')
    ndim = len(shape)
    lead = x.ndim - ndim
    lead_axis = tuple(range(lead))
    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y = x.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        y = y.squeeze(lead_axis)
    return y


class Sum(Operator):
    """
    This class is used to sum a tensor along a given axis.
    """

    def __init__(self, axis=None, keepdims=False):
        super().__init__()
        self.x_shape = None
        self.axis = axis
        self.keepdims = keepdims

    @property
    def label(self):
        return '__sum__'

    def forward(self, x):
        self.x_shape = x.shape
        if isinstance(self.axis, list):
            self.axis = tuple(self.axis)
            # self.axis = self.axis[0]
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, grad_output):
        gy = grad_output
        ndim = len(self.x_shape)
        tuple_axis = self.axis
        if self.axis is None:
            tuple_axis = None
        elif not isinstance(self.axis, tuple):
            tuple_axis = (self.axis,)

        if not (ndim == 0 or tuple_axis is None or self.keepdims):
            actual_axis = [a if a >= 0 else a + ndim for a in tuple_axis] # convert negative axis to positive axis, Neil 2020-09-16
            shape = list(gy.shape)
            for a in sorted(actual_axis):
                shape.insert(a, 1)
        else:
            shape = gy.shape

        gy = gy.reshape(shape)

        grad_x = broadcast_to(gy, self.x_shape)
        return grad_x


def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)


class BroadcastTo(Operator):
    """
    This class is used to broadcast a tensor to a given shape.
    """

    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        self.x_shape = None

    @property
    def label(self):
        return '__broadcast_to__'

    def forward(self, x):
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y

    def backward(self, grad_output):
        grad_x = sum_to(grad_output, self.x_shape)
        return grad_x


def broadcast_to(x, shape):
    """
    This function is used to broadcast a tensor to a given shape.
    :param x:
    :param shape:
    :return:
    """
    if x.shape == shape:
        return as_tensor(x)
    return BroadcastTo(shape)(x)


class Index(Operator):
    """
    This class is used to get a slice of a tensor.
    """

    def __init__(self, slices):
        super().__init__()
        self.x_shape = None
        if not isinstance(slices, tuple):
            slices = (slices,)
        self.slices = slices

    @property
    def label(self):
        return '__index__'

    def forward(self, x):
        self.x_shape = x.shape
        y = x[self.slices]
        return y

    def backward(self, grad_output):
        ig = IndexGrad(self.slices, self.x_shape)
        return ig(grad_output)


def index(x, slices):
    return Index(slices)(x)


class IndexGrad(Operator):
    """
    This class is used to compute the gradient of index operator.
    """

    def __init__(self, slices, x_shape):
        super().__init__()
        self.slices = slices
        self.x_shape = x_shape

    @property
    def label(self):
        return '__index_grad__'

    def forward(self, grad_output):
        grad_x = np.zeros(self.x_shape, dtype=grad_output.dtype)
        np.add.at(grad_x, self.slices, grad_output)
        return grad_x

    def backward(self, grad_of_grad_output):
        return index(grad_of_grad_output, self.slices)


class Log(Operator):
    """
    This class is used to compute the logarithm of a tensor.
    """
    @property
    def label(self):
        return '__log__'

    def forward(self, x):
        y = np.log(x)
        return y

    def backward(self, grad_output):
        x = self.inputs[0]
        grad_x = grad_output / x
        return grad_x


class Clip(Operator):
    """
    This class is used to clip the value of a tensor.
    """
    @property
    def label(self):
        return '__clip__'

    def __init__(self, a_min, a_max):
        super().__init__()
        self.a_min = a_min
        self.a_max = a_max

    def forward(self, x):
        y = np.clip(x, self.a_min, self.a_max)
        return y

    def backward(self, grad_output):
        x, = self.inputs
        mask = (x.data >= self.a_min) & (x.data <= self.a_max)
        grad_x = grad_output * mask
        return grad_x


def clip(x, a_min, a_max):
    return Clip(a_min, a_max)(x)


def log(x):
    return Log()(x)


class Concat(Operator):
    """
    This class is used to concatenate tensors.
    """
    def __init__(self, axis):
        super().__init__()
        self.axis = axis

    @property
    def label(self):
        return '__concat__'

    def forward(self, *xs):
        y = np.concatenate(xs, axis=self.axis)
        return y

    def backward(self, grad_output):
        xs = self.inputs
        axis = self.axis
        split_sizes = [x.shape[axis] for x in xs]
        for (i, size) in enumerate(split_sizes):
            if i > 0:
                split_sizes[i] += split_sizes[i - 1]
        grads = np.split(grad_output.data, split_sizes, axis=axis)
        ret_grads = []
        for i in range(len(xs)):
            ret_grads.append(grads[i])
        return ret_grads


def concat(*xs, axis=1):
    return Concat(axis)(*xs)


def init_tensor():
    Tensor.__array__priority__ = 100
    Tensor.__add__ = add
    Tensor.__radd__ = radd
    Tensor.__sub__ = sub
    Tensor.__rsub__ = rsub
    Tensor.__mul__ = mul
    Tensor.__rmul__ = rmul
    Tensor.__truediv__ = div
    Tensor.__rtruediv__ = rdiv
    Tensor.__neg__ = neg
    Tensor.__abs__ = abs
    Tensor.__pow__ = pow
    Tensor.__matmul__ = matmul
    Tensor.dot = matmul
    Tensor.__getitem__ = index
