from pinenut.core import Parameter, Tensor, matmul, softmax_cross_entropy, accuracy
import pinenut.core.dataset as ds
import pinenut.core as C
import pinenut.core.dropout as DO
import numpy as np
import os


class LayerBase:
    def __init__(self):
        self._params = set()
        self._children = set()
        self.name = None

    def __setattr__(self, key, value):
        if isinstance(value, Parameter):
            self._params.add(key)
        elif isinstance(value, LayerBase):
            self._children.add(key)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        if item in self._params:
            self._params.remove(item)
        elif item in self._children:
            self._children.remove(item)
        super().__delattr__(item)

    def __call__(self, *inputs):
        return self.forward(*inputs)

    def forward(self, *inputs):
        raise NotImplementedError

    @property
    def params(self):
        d = self.__dict__
        for name in self._params:
            yield d[name]
        for name in self._children:  # recursively access the params of children
            yield from d[name].params

    def clear_all_grad(self):
        for param in self.params:
            param.clear_grad()
        for name in self._children:  # recursively clear the grad of children
            self.__dict__[name].clear_all_grad()

    def unwrap_tensor(self, obj):
        if isinstance(obj, Tensor):
            return obj.data
        else:
            return obj

    def _params_and_path(self, all_params_dict, current_path=''):
        d = self.__dict__
        for name in self._params:
            key = current_path + '.' + name  # build a full path for each parameter, in order to uniquely identify it
            all_params_dict[key] = self.unwrap_tensor(d[name])
        for name in self._children:  # recursively access the params of children
            key = current_path + '.' + name
            d[name]._params_and_path(all_params_dict, key)

    def save_weights(self, path):
        all_params_dict = {}
        self._params_and_path(all_params_dict, '')

        # only save parameters which is not None
        for k, param in all_params_dict.items():
            if param is None:
                all_params_dict.pop(k)

        try:
            np.savez(path, **all_params_dict, allow_pickle=True)
            # np.savez_compressed(path, **all_params_dict, allow_pickle=True)
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(path):
                os.remove(path)
            print(e)
            print('save weights failed.')

    def load_weights(self, path):
        """
        Load parameters from a npz file.
        Before load_params() you should call init_model_weight() to
        initialize the weight data structure.
        :param path:
        :return:
        """
        # print('loading weights from %s' % path)
        try:
            from_npz = np.load(path, allow_pickle=True)
            all_params_dict = {}
            self._params_and_path(all_params_dict, '')
            for k, param in all_params_dict.items():
                param.data = from_npz[k]
        except (Exception, KeyboardInterrupt) as e:
            print('load weights failed.')
            print('before load_weights() you should call init_model_weight() to initialize the weight data structure.')
            raise e


class Linear(LayerBase):
    def __init__(self, out_features, in_features=None, dtype=np.float64, bias=True, activation=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        self.activation = activation
        self.weight = Parameter(None, name='weight')
        if self.in_features is not None:
            self.init_weight()
        if bias:
            self.bias = Parameter(np.zeros(out_features, dtype=dtype), name='bias')
        else:
            self.bias = None

    def init_weight(self):
        self.weight.data = np.random.randn(self.in_features, self.out_features).astype(self.dtype) * np.sqrt(
            2 / self.in_features)

    def forward(self, x):
        if self.weight.data is None:
            self.in_features = x.shape[-1]
            self.init_weight()

        y = matmul(x, self.weight)
        # y = x @ self.weight

        if self.bias is not None:
            y += self.bias
        return y


class MLP(LayerBase):
    def __init__(self, layer_output_sizes=(), in_features=None, hidden_activation=None, output_activation=None):
        super().__init__()
        self.layers = []
        layer_count = len(layer_output_sizes)
        for i, size in enumerate(layer_output_sizes):
            if i < layer_count - 1:
                activation = hidden_activation
            else:
                activation = output_activation
            layer = Linear(out_features=size, in_features=in_features, activation=activation)
            setattr(self, 'linear%d' % i, layer)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            # x = DO.dropout(x, 0.1)
            if layer.activation is not None:
                x = layer.activation(x)
        return x

    def add_layer(self, layer):
        self.layers.append(layer)
        setattr(self, 'linear%d' % len(self.layers), layer)

    def init_model_weight(self, in_features=None):
        """
        Initialize the weight of each layer, before load_params() you should call this function to initialize the weight
        :param in_features:
        :return:
        """
        input_size = in_features
        for layer in self.layers:
            layer.in_features = input_size
            layer.init_weight()
            input_size = layer.out_features

    def summary(self):
        print('---------------------------------------------------------------')
        print('Layer (type)             Output Shape              Activation #')
        print('===============================================================')

        for i, layer in enumerate(self.layers):
            name = 'linear%d' % i
            output_shape = layer.out_features
            if layer.activation is not None:
                describe = layer.activation.__name__
            else:
                describe = 'None'
            print('{:<30} {:<20} {:<25}'.format(name, str(output_shape), describe))
        print('===============================================================')
        print('Total layers: %d' % len(self.layers))
        print('---------------------------------------------------------------')

    def train(self, train_dataset, epochs, batch_size, optimizer, test_dataset=None):
        train_loader = ds.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            sum_loss = 0.0
            sum_acc = 0.0
            for x, y in train_loader:
                y_pred = self(x)
                loss = softmax_cross_entropy(y_pred, y)

                self.clear_all_grad()
                loss.backward()
                optimizer.update()

                acc = accuracy(y_pred, y)
                sum_loss += float(loss.data) * len(y)
                sum_acc += float(acc.data) * len(y)

            train_loader.reset()
            print('epoch:{} '.format(epoch + 1))
            tl = sum_loss / len(train_dataset)
            print('train loss: {:.4f}, accuracy:{:.4f}'
                  .format(sum_loss / len(train_dataset), sum_acc / len(train_dataset)))

            if test_dataset is not None:
                with C.no_grad():
                    test_loader = ds.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
                    sum_loss = 0.0
                    sum_acc = 0.0
                    for x, y in test_loader:
                        y_pred = self.predict(x)
                        loss = softmax_cross_entropy(y_pred, y)

                        acc = accuracy(y_pred, y)
                        sum_loss += float(loss.data) * len(y)
                        sum_acc += float(acc.data) * len(y)

                    test_loader.reset()
                    print('test loss: {:.4f}, accuracy:{:.4f}'
                          .format(sum_loss / len(test_dataset), sum_acc / len(test_dataset)))
            print('---------------------------------------------------------------')

    def predict(self, *inputs):
        return self.forward(*inputs)


class Embedding(LayerBase):
    def __init__(self, num_embeddings, embedding_dim, dtype=np.float64):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.dtype = dtype
        self.weight = Parameter(np.random.randn(num_embeddings, embedding_dim).astype(dtype) * np.sqrt(
            2 / embedding_dim), name='weight')

    def forward(self, x):
        return self.weight[x]

