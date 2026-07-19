import gzip

from pinenut.core import Dataset
import numpy as np
import pinenut.core.utils as U


class SpiralDataset(Dataset):
    def __init__(self, n_samples=100, n_classes=3, n_dim=2, xdatatype=np.float64, ydatatype=np.int32):
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.n_dim = n_dim
        self.xdatatype = xdatatype
        self.ydatatype = ydatatype
        super().__init__()

    def generate(self):
        data_size = self.n_samples * self.n_classes
        X = np.zeros((data_size, self.n_dim), dtype=self.xdatatype)
        y = np.zeros(data_size, dtype=self.ydatatype)

        for j in range(self.n_classes):
            for i in range(self.n_samples):
                rate = i / self.n_samples
                radius = 1.0 * rate
                theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.5
                ix = self.n_samples * j + i
                X[ix] = np.array([radius * np.sin(theta),
                                  radius * np.cos(theta)]).flatten()
                y[ix] = j

        # shuffle
        indices = np.arange(self.n_samples * self.n_classes)
        np.random.shuffle(indices)
        self.data = X[indices]
        self.label = y[indices]

    def load_data(self):
        self.generate()


class MNIST(Dataset):
    def __init__(self, train=True, data_transform=None, label_transform=None, xdatatype=np.float32, ydatatype=np.int32):
        self.xdatatype = xdatatype
        self.ydatatype = ydatatype
        super().__init__(train=train, data_transform=data_transform, label_transform=label_transform)

    def load_data(self):
        train_urls = ['http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
                        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz']
        test_urls = ['http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
                        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz']

        urls = train_urls if self.train else test_urls
        data_file = U.download(urls[0]) # download the data file
        label_file = U.download(urls[1]) # download the label file

        # load the data file
        with gzip.open(data_file, 'rb') as f:
            X = np.frombuffer(f.read(), np.uint8, offset=16)
            X = X.reshape(-1, 1, 28, 28)
            self.data = X

        # load the label file
        with gzip.open(label_file, 'rb') as f:
            y = np.frombuffer(f.read(), np.uint8, offset=8)
            self.label = y

    def show(self, index):
        import matplotlib.pyplot as plt
        plt.imshow(self.data[index].reshape(28, 28), cmap='gray')
        plt.show()


if __name__ == '__main__':
    dataset = MNIST(train=True)
    X = dataset.data
    y = dataset.label
    print(X.shape)
    print(y.shape)
    dataset.show(0)