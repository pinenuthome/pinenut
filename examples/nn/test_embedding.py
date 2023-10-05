from pinenut import Embedding
import unittest
import pinenut.core.cuda as cuda


class TestEmbedding(unittest.TestCase):
    def test_lookup(self):
        embedding = Embedding(10, 5)
        if cuda.Cuda.available():
            embedding.to_gpu()
  
        x = xp.array([1, 2, 5])
        y = embedding(x)
        assert y.shape == (3, 5)
        y.backward()
        grad = embedding.weight.grad[x]
        assert grad == xp.ones_like(y.data)

    def test_embedding_value_change(self):
        embedding = Embedding(5, 2)
        if cuda.Cuda.available():
            embedding.to_gpu()
       
        x = xp.array([1, 2])
        y = embedding(x)

        print("if origin value has been changed")
        y.data[0][0] = 100

        print(y)
        print(embedding.weight)
        assert embedding.weight[1][0] != 100


if __name__ == '__main__':
    xp = cuda.Cuda.xp() if cuda.Cuda.available() else cuda.Cuda.numpy()
    unittest.main()
