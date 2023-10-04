from pinenut import Embedding
import numpy as np
import unittest


class TestEmbedding(unittest.TestCase):
    def test_lookup(self):
        embedding = Embedding(10, 5)
        print(embedding.weight)

        x = np.array([1, 2, 5])
        y = embedding(x)
        assert y.shape == (3, 5)

        y.backward()
        grad = embedding.weight.grad[x]
        assert grad == np.ones_like(y)

    def test_embedding_value_change(self):
        embedding = Embedding(5, 2)
        print(embedding.weight)

        x = np.array([1, 2])
        y = embedding(x)
        print(y)

        print("if origin value has been changed")
        y.data[0][0] = 100

        print(y)
        print(embedding.weight)
        assert embedding.weight[1][0] != 100


if __name__ == '__main__':
    unittest.main()