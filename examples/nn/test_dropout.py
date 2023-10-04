import numpy as np

import pinenut.core.dropout as dp
import pinenut.core as C
from pinenut.core import Tensor
import unittest


class TestDropout(unittest.TestCase):
    def test_dropout(self):
        x = Tensor(np.ones(10))
        y = dp.dropout(x, 0.3)
        p1 = np.sum(y.data) == 7
        p2 = np.sum(y.data) == 8
        assert p1 or p2
        print(y)

        y.backward()
        assert x.grad == y.data
        print(x.grad)

        with C.no_train():
            y = dp.dropout(x, 0.3)
            assert y == x
            print(y)


if __name__ == '__main__':
    unittest.main()