from pinenut import Tensor
import unittest


class TestComplexMath(unittest.TestCase):
    def test_goldstein_price(self):
        x = Tensor(0.0)
        y = Tensor(-1.0)
        z = (1 + (x + y + 1)**2 * (19 - 14 * x + 3 * x**2 - 14 * y + 6 * x * y + 3 * y**2)) * \
            (30 + (2 * x - 3 * y)**2 * (18 - 32 * x + 12 * x**2 + 48 * y - 36 * x * y + 27 * y**2))
        z.backward()

        print('z=', z)
        print('x.grad=', x.grad)
        print('y.grad=', y.grad)

        self.assertAlmostEqual(z.data, 3.0, places=5)
        self.assertAlmostEqual(x.grad, 0.0, places=5)
        self.assertAlmostEqual(y.grad, 0.0, places=5)

    def test_rosenbrock(self):
        x = Tensor(1.0)
        y = Tensor(1.0)
        z = (x - 1)**2 + 100 * (y - x**2)**2
        z.backward()

        print('z=', z)
        print('x.grad=', x.grad)
        print('y.grad=', y.grad)

        self.assertAlmostEqual(z.data, 0.0, places=5)
        self.assertAlmostEqual(x.grad, 0.0, places=5)
        self.assertAlmostEqual(y.grad, 0.0, places=5)


if __name__ == '__main__':
    unittest.main()