import unittest

import numpy as np

import pinenut as pn
from pinenut import nn


class TestCuda(unittest.TestCase):
    def test_cuda(self):
        x = pn.Tensor([1, 2, 3])
        y = pn.Tensor([4, 5, 6])

        if pn.cuda.is_available():
            x.cuda()
            y.cuda()

        result = pn.matmul(x, y.T)
        self.assertEqual(pn.cuda.as_numpy(result.data).item(), 32.0)
        result.backward()

        xp = pn.cuda.get_array_module(x.data)
        self.assertTrue(xp.array_equal(x.grad.data, xp.array([4, 5, 6])))
        self.assertTrue(xp.array_equal(
            y.grad.data.squeeze(), xp.array([1, 2, 3])))

    def test_tensor_device_round_trip(self):
        tensor = pn.Tensor(np.array([1.0, 2.0]))
        self.assertIs(tensor.to('cpu'), tensor)
        self.assertIsInstance(tensor.data, np.ndarray)

    @unittest.skipUnless(pn.cuda.is_available(), 'CUDA is not available')
    def test_tensor_moves_existing_gradient_with_data(self):
        tensor = pn.tensor(np.array([1.0, 2.0]))
        pn.sum(tensor ** 2).backward()

        tensor.cuda(0)

        cp = pn.cuda.cupy()
        self.assertIsInstance(tensor.data, cp.ndarray)
        self.assertIsInstance(tensor.grad.data, cp.ndarray)
        self.assertEqual(tensor.data.device.id, tensor.grad.data.device.id)

        tensor.cpu()
        self.assertIsInstance(tensor.data, np.ndarray)
        self.assertIsInstance(tensor.grad.data, np.ndarray)

    @unittest.skipUnless(pn.cuda.is_available(), 'CUDA is not available')
    def test_reset_parameters_stays_on_gpu(self):
        layer = nn.Linear(3, 2).cuda(0)

        layer.reset_parameters()

        for parameter in layer.parameters():
            self.assertEqual(parameter.device, 'cuda:0')

    @unittest.skipUnless(pn.cuda.is_available(), 'CUDA is not available')
    def test_lazy_state_dict_load_stays_on_gpu(self):
        layer = nn.LazyLinear(2).cuda(0)
        state = {
            'bias': np.zeros(2),
            'weight': np.ones((2, 3)),
        }

        layer.load_state_dict(state)

        self.assertEqual(layer.weight.device, 'cuda:0')
        self.assertEqual(layer.bias.device, 'cuda:0')


if __name__ == '__main__':
    unittest.main()
