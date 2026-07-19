import numpy as np

import pinenut as pn
from pinenut import nn
import unittest


class TestDropout(unittest.TestCase):
    def test_dropout(self):
        np.random.seed(0)
        layer = nn.Dropout(0.3)
        x = pn.Tensor(np.ones(1000))
        y = layer(x)
        assert np.isin(y.data, [0, 1 / 0.7]).all()
        assert abs(y.data.mean() - 1.0) < 0.1

        y.backward()
        assert x.grad == y.data

        x.zero_grad()
        layer.eval()
        y = layer(x)
        assert y == x
        y.backward()
        assert x.grad == np.ones_like(x.data)

    def test_invalid_probability(self):
        with self.assertRaises(ValueError):
            nn.Dropout(1.0)


if __name__ == '__main__':
    unittest.main()
