from pinenut import Tensor, build_graph
from pinenut.core.computational_graph import ComputationalGraph
import shutil
import tempfile
import unittest
from pathlib import Path


class TestBuildAGraph(unittest.TestCase):
    def test_default_filename(self):
        graph = ComputationalGraph(Tensor(1.0))
        self.assertEqual(graph.dot.filename.name, 'computational_graph.png')

    def test_grad(self):
        if shutil.which('dot') is None:
            self.skipTest('Graphviz is not installed')

        x = Tensor(3)
        y = Tensor(4)
        z = x * y + x**2

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'test_cp.png'
            build_graph(z, to_file=output, verbose=True, view=False)
            self.assertTrue(output.exists())
            self.assertTrue(Path(str(output) + '.dot').exists())


if __name__ == '__main__':
    unittest.main()
