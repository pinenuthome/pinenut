import unittest

import pinenut as pn
from pinenut import nn

xp = pn.cuda.xp()


class TestEmbedding(unittest.TestCase):
    def test_lookup(self):
        embedding = nn.Embedding(10, 5)
        if pn.cuda.is_available():
            embedding.cuda()
  
        x = xp.array([1, 2, 5])
        y = embedding(x)
        assert y.shape == (3, 5)
        y.backward()
        grad = embedding.weight.grad[x]
        assert grad == xp.ones_like(y.data)

    def test_embedding_value_change(self):
        embedding = nn.Embedding(5, 2)
        if pn.cuda.is_available():
            embedding.cuda()
       
        x = xp.array([1, 2])
        y = embedding(x)

        y.data[0][0] = 100

        assert embedding.weight[1][0] != 100


if __name__ == '__main__':
    unittest.main()
