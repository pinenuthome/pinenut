import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np

import pinenut as pn
from pinenut import datasets, nn, optim
from pinenut.nn import functional as F
from pinenut.utils.data import DataLoader


class TinyDataset:
    def __init__(self, size=4):
        self.data = np.arange(size * 2).reshape(size, 2)
        self.labels = np.arange(size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


class TestBackwardRegressions(unittest.TestCase):
    def test_same_graph_can_run_backward_twice(self):
        x = pn.Tensor(3.0)
        y = x * 2

        y.backward()
        self.assertEqual(x.grad, 2.0)

        x.zero_grad()
        y.backward()
        self.assertEqual(x.grad, 2.0)

    def test_repeated_backward_accumulates_leaf_gradient(self):
        x = pn.Tensor(3.0)
        y = x * 2

        y.backward()
        y.backward()

        self.assertEqual(x.grad, 4.0)

    def test_backward_engine_does_not_store_graph_state(self):
        self.assertFalse(hasattr(pn.Tensor._backward_engine, 'ops'))
        self.assertFalse(hasattr(pn.Tensor._backward_engine, 'op_set'))

    def test_concat_returns_tensor_gradients(self):
        x0 = pn.Tensor([1.0, 2.0])
        x1 = pn.Tensor([3.0, 4.0])

        pn.concat(x0, x1, axis=0).backward()

        self.assertIsInstance(x0.grad, pn.Tensor)
        self.assertIsInstance(x1.grad, pn.Tensor)

    def test_matmul_matrix_vector_backward(self):
        matrix = pn.tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        vector = pn.tensor(np.array([5.0, 6.0]))

        pn.sum(pn.matmul(matrix, vector)).backward()

        np.testing.assert_array_equal(
            matrix.grad.data, np.array([[5.0, 6.0], [5.0, 6.0]]))
        np.testing.assert_array_equal(vector.grad.data, np.array([4.0, 6.0]))

    def test_matmul_batched_backward_preserves_input_shapes(self):
        left = pn.tensor(np.ones((2, 3, 4)))
        right = pn.tensor(np.ones((4, 5)))

        pn.sum(pn.matmul(left, right)).backward()

        self.assertEqual(left.grad.shape, left.shape)
        self.assertEqual(right.grad.shape, right.shape)
        np.testing.assert_array_equal(left.grad.data, np.full(left.shape, 5.0))
        np.testing.assert_array_equal(right.grad.data, np.full(right.shape, 6.0))


class TestModuleAndOptimizerRegressions(unittest.TestCase):
    def test_adam_increments_once_per_step(self):
        layer = nn.Linear(3, 2)
        original = {
            name: parameter.data.copy()
            for name, parameter in layer.named_parameters()
        }
        for parameter in layer.parameters():
            parameter.grad = pn.Tensor(np.ones_like(parameter.data))

        optimizer = optim.Adam(layer.parameters())
        optimizer.step()

        self.assertEqual(optimizer.step_count, 1)
        for name, parameter in layer.named_parameters():
            np.testing.assert_allclose(
                parameter.data, original[name] - optimizer.lr)

    def test_adam_does_not_increment_without_gradients(self):
        layer = nn.Linear(3, 2)
        optimizer = optim.Adam(layer.parameters())
        optimizer.step()
        self.assertEqual(optimizer.step_count, 0)

    def test_optimizer_zero_grad(self):
        layer = nn.Linear(3, 2)
        optimizer = optim.SGD(layer.parameters())
        for parameter in layer.parameters():
            parameter.grad = pn.Tensor(np.ones_like(parameter.data))

        optimizer.zero_grad()

        self.assertTrue(all(
            parameter.grad is None for parameter in layer.parameters()))

    def test_shared_parameters_are_returned_and_updated_once(self):
        shared = nn.Linear(1, 1, bias=False)
        model = nn.Sequential(shared, shared)
        shared.weight.data.fill(1.0)
        shared.weight.grad = pn.tensor(np.ones_like(shared.weight.data))

        named_parameters = list(model.named_parameters())
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        optimizer.step()

        self.assertEqual(len(named_parameters), 1)
        self.assertEqual(named_parameters[0][0], '0.weight')
        self.assertEqual(len(optimizer.params), 1)
        np.testing.assert_allclose(shared.weight.data, [[0.9]])

    def test_optimizer_deduplicates_explicit_parameter_list(self):
        parameter = nn.Parameter(np.array([1.0]))
        parameter.grad = pn.tensor(np.array([1.0]))
        optimizer = optim.SGD([parameter, parameter], lr=0.1)

        optimizer.step()

        self.assertEqual(len(optimizer.params), 1)
        np.testing.assert_allclose(parameter.data, [0.9])

    def test_mlp_propagates_layer_sizes(self):
        model = nn.MLP(3, (4, 2))
        output = model(pn.Tensor(np.ones((5, 3))))

        self.assertEqual(output.shape, (5, 2))
        self.assertEqual(model.linear0.weight.shape, (4, 3))
        self.assertEqual(model.linear1.weight.shape, (2, 4))

    def test_state_dict_round_trip(self):
        np.random.seed(0)
        model = nn.MLP(3, (4, 2))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'weights.npz'
            pn.save(model.state_dict(), path)
            restored = nn.MLP(3, (4, 2))
            restored.load_state_dict(pn.load(path))

        for (name, expected), (restored_name, actual) in zip(
                model.named_parameters(), restored.named_parameters()):
            self.assertEqual(name, restored_name)
            np.testing.assert_array_equal(actual.data, expected.data)

    def test_state_dict_rejects_uninitialized_parameters(self):
        model = nn.LazyLinear(2)
        with self.assertRaisesRegex(ValueError, 'uninitialized'):
            model.state_dict()

    def test_train_and_eval_propagate_to_children(self):
        model = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.5))
        model.eval()
        self.assertFalse(model.training)
        self.assertFalse(getattr(model, '1').training)

        model.train()
        self.assertTrue(model.training)
        self.assertTrue(getattr(model, '1').training)

    def test_mlp_fit_runs_training_loop(self):
        dataset = datasets.SpiralDataset(n_samples=5, n_classes=3)
        model = nn.MLP(2, (4, 3), hidden_activation=F.relu)
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        with redirect_stdout(StringIO()):
            result = model.fit(
                dataset, epochs=1, batch_size=5, optimizer=optimizer)

        self.assertIs(result, model)


class TestNumericalAndDataRegressions(unittest.TestCase):
    def test_softmax_is_stable_for_large_logits(self):
        logits = pn.Tensor(np.array([[1000.0, 1001.0]]))
        probabilities = F.softmax(logits)

        self.assertTrue(np.isfinite(probabilities.data).all())
        np.testing.assert_allclose(probabilities.data.sum(axis=1), [1.0])

    def test_cross_entropy_keeps_gradient_for_extreme_wrong_logit(self):
        logits = pn.tensor(np.array([[1000.0, -1000.0]]))

        loss = F.cross_entropy(logits, np.array([1]))
        loss.backward()

        self.assertAlmostEqual(float(loss.data), 2000.0)
        np.testing.assert_allclose(logits.grad.data, [[1.0, -1.0]])

    def test_binary_cross_entropy_with_logits_is_stable(self):
        logits = pn.tensor(np.array([[-1000.0, 1000.0]]))
        targets = pn.tensor(np.array([[1.0, 0.0]]))

        loss = F.binary_cross_entropy_with_logits(logits, targets)
        loss.backward()

        self.assertTrue(np.isfinite(loss.data))
        self.assertTrue(np.isfinite(logits.grad.data).all())
        self.assertAlmostEqual(float(loss.data), 1000.0)
        np.testing.assert_allclose(logits.grad.data, [[-0.5, 0.5]])

    def test_binary_cross_entropy_accepts_boundary_probabilities(self):
        probabilities = pn.tensor(np.array([0.0, 1.0], dtype=np.float32))
        targets = pn.tensor(np.array([0.0, 1.0], dtype=np.float32))

        loss = F.binary_cross_entropy(probabilities, targets)
        loss.backward()

        self.assertTrue(np.isfinite(loss.data))
        self.assertTrue(np.isfinite(probabilities.grad.data).all())

    def test_elementwise_losses_average_all_elements(self):
        zeros = pn.tensor(np.zeros((2, 3)))
        ones = pn.tensor(np.ones((2, 3)))
        probabilities = pn.tensor(np.full((2, 3), 0.5))

        mse = F.mse_loss(zeros, ones)
        bce = F.binary_cross_entropy(probabilities, ones)

        self.assertAlmostEqual(float(mse.data), 1.0)
        self.assertAlmostEqual(float(bce.data), np.log(2.0))

    def test_mse_accepts_scalar_inputs(self):
        loss = F.mse_loss(pn.tensor(0.0), pn.tensor(1.0))
        self.assertAlmostEqual(float(loss.data), 1.0)

    def test_data_loader_initializes_device_state(self):
        loader = DataLoader(TinyDataset(), batch_size=2)
        data, labels = next(loader)

        self.assertEqual(loader.device, 'cpu')
        self.assertIsNone(loader.device_list)
        self.assertEqual(data.shape, (2, 2))
        self.assertEqual(labels.shape, (2,))

    def test_data_loader_can_drop_last_batch(self):
        loader = DataLoader(TinyDataset(size=5), batch_size=2, drop_last=True)
        batches = list(loader)

        self.assertEqual(len(batches), 2)
        self.assertEqual(sum(len(data) for data, _ in batches), 4)

    def test_data_loader_rejects_invalid_batch_size(self):
        for value in (0, -1):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    DataLoader(TinyDataset(), batch_size=value)
        for value in (True, 1.5):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    DataLoader(TinyDataset(), batch_size=value)

    def test_dataset_rejects_invalid_indexes(self):
        dataset = datasets.SpiralDataset(n_samples=1, n_classes=2)

        with self.assertRaises(IndexError):
            _ = dataset[-1]
        with self.assertRaises(IndexError):
            _ = dataset[len(dataset)]
        with self.assertRaises(TypeError):
            _ = dataset[0.5]

    def test_dropout_eval_backward_is_identity(self):
        dropout = nn.Dropout(0.5).eval()
        x = pn.Tensor(np.ones(4))
        y = dropout(x)
        y.backward()

        np.testing.assert_array_equal(y.data, x.data)
        np.testing.assert_array_equal(x.grad.data, np.ones(4))

    def test_dropout_rejects_probability_one(self):
        with self.assertRaises(ValueError):
            nn.Dropout(1.0)


class TestDataParallelRegressions(unittest.TestCase):
    def test_requires_two_distinct_devices(self):
        with self.assertRaises(ValueError):
            nn.DataParallel([0])
        with self.assertRaises(ValueError):
            nn.DataParallel([0, 0])

    def test_training_requires_cupy(self):
        if pn.cuda.is_available():
            self.skipTest('CUDA is available')

        model = nn.DataParallel([0, 1], layer_sizes=(2,))
        with self.assertRaises(RuntimeError):
            model.fit(TinyDataset(), epochs=1, batch_size=2)


class TestPublicAPI(unittest.TestCase):
    def test_removed_names_are_not_exported(self):
        for name in ('Cuda', 'LayerBase', 'ParallelModel', 'AdaGrad',
                     'Momentum', 'mean_squared_error',
                     'softmax_cross_entropy', 'no_train'):
            self.assertFalse(hasattr(pn, name), name)

        self.assertFalse(hasattr(nn.Module, 'clear_all_grad'))
        self.assertFalse(hasattr(nn.Module, 'named_params'))
        self.assertFalse(hasattr(pn.Tensor, 'clear_grad'))
        self.assertFalse(hasattr(pn.Tensor, 'to_gpu'))
        self.assertFalse(hasattr(optim.Optimizer, 'update'))


if __name__ == '__main__':
    unittest.main()
