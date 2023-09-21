from pinenut import Tensor
import unittest


class TestAutograd(unittest.TestCase):
    def test_grad(self):
        x = Tensor(3)
        y = Tensor(4)
        z = x * y
        z.backward()
        self.assertEqual(x.grad.data, 4)
        self.assertEqual(y.grad.data, 3)

    def test_second_order_gradient(self):
        x = Tensor(3)
        y = x ** 2
        y.backward()
        gx = x.grad
        self.assertEqual(gx.data, 6)
        x.clear_grad()
        gx.backward()
        self.assertEqual(x.grad, 2)


if __name__ == '__main__':
    unittest.main()