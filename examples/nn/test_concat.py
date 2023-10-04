import numpy as np
from pinenut import Tensor, concat
import unittest


class TestConcat(unittest.TestCase):
    def test_concat(self):
        x1 = Tensor([1, 2, 3])
        x2 = Tensor([4, 5, 6])
        x3 = Tensor([7, 8, 9])
        y = concat(*[x1, x2, x3], axis=0)
        assert y == [1, 2, 3, 4, 5, 6, 7, 8, 9]

        x1 = Tensor([[1, 2, 3], [4, 5, 6]])
        x2 = Tensor([[7, 8, 9], [10, 11, 12]])
        y = concat(*[x1, x2], axis=0)
        assert y == [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]

        y = concat(*[x1, x2], axis=1)
        assert y == [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]

        x1 = Tensor([[[1, 2, 3], [4, 5, 6]]])
        x2 = Tensor([[[7, 8, 9], [10, 11, 12]]])
        y = concat(*[x1, x2], axis=0)
        assert y == [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]

        y = concat(*[x1, x2], axis=1)
        assert y == [[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]]

        y = concat(*[x1, x2], axis=2)
        assert y == [[[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]]

    def test_concat_backward(self):
        x1 = Tensor([1, 2, 3])
        x2 = Tensor([4, 5, 6])
        x3 = Tensor([7, 8, 9])
        y = concat(*[x1, x2, x3], axis=0)
        assert y == [1, 2, 3, 4, 5, 6, 7, 8, 9]

        y.backward()
        print(x1.grad == np.array([1, 1, 1]))
        assert (x1.grad == np.array([1, 1, 1])).all()
        assert (x2.grad == np.array([1, 1, 1])).all()
        assert (x3.grad == np.array([1, 1, 1])).all()


if __name__ == '__main__':
    unittest.main()