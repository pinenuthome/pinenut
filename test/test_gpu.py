from pinenut import Tensor, Cuda, matmul, as_array
import unittest


class TestCuda(unittest.TestCase):
    def test_cuda(self):
        x = Tensor([1, 2, 3])
        y = Tensor([4, 5, 6])

        cuda_is_available = Cuda.available()
        if cuda_is_available:
            x.to_gpu()
            y.to_gpu()

        z = matmul(x, y.T)

        assert z.data == as_array(32)
        print(type(z.data))

        z.backward()

        xp = Cuda.get_array_module(x.data)
        assert xp.array_equal(x.grad.data, xp.array([4, 5, 6]))
        assert xp.array_equal(y.grad.data.squeeze(), xp.array([1, 2, 3]))

        # assert (x.grad.data == [4, 5, 6]).all()
        # assert (y.grad.data == [1, 2, 3]).all()


if __name__ == '__main__':
    unittest.main()