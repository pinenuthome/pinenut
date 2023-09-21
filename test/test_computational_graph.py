from pinenut import Tensor, build_graph
import unittest


class TestBuildAGraph(unittest.TestCase):
    def test_grad(self):
        x = Tensor(3)
        y = Tensor(4)
        z = x * y + x**2

        build_graph(z, to_file='test_cp.png', verbose=True, view=True)


if __name__ == '__main__':
    unittest.main()