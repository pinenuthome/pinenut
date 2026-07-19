from pinenut import Tensor
import unittest


class TestBasicOps(unittest.TestCase):
    def test_add(self):
        a = Tensor(2.0)
        b = Tensor(3.0)
        c = a + b
        self.assertEqual(c, 5.0)
        c.backward()
        self.assertEqual(a.grad, 1.0)

    def test_sub(self):
        a = Tensor(2.0)
        b = Tensor(3.0)
        c = a - b
        self.assertEqual(c, -1.0)
        c.backward()
        self.assertEqual(a.grad, 1.0)

    def test_mul(self):
        a = Tensor(2.0)
        b = Tensor(3.0)
        c = a * b
        self.assertEqual(c, 6.0)
        c.backward()
        self.assertEqual(a.grad, 3.0)

    def test_div(self):
        a = Tensor(2.0)
        b = Tensor(3.0)
        c = a / b
        self.assertEqual(c, 2.0 / 3.0)
        c.backward()
        self.assertEqual(a.grad, 1.0 / 3.0)

    def test_pow(self):
        a = Tensor(2.0)
        b = Tensor(3.0)
        c = a ** b
        self.assertEqual(c, 8.0)
        c.backward()
        self.assertEqual(a.grad, 12.0)


if __name__ == '__main__':
    unittest.main()