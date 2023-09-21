import numpy as np


class Dataset:
    def __init__(self, train=True, data_transform=None, label_transform=None):
        self.train = train
        self.data = None
        self.label = None
        self.data_transform = data_transform
        self.label_transform = label_transform
        if self.data_transform is None:
            self.data_transform = lambda x: x
        if self.label_transform is None:
            self.label_transform = lambda x: x
        self.load_data()

    def __getitem__(self, item):
        if not (np.isscalar(item) and 0 <= item < len(self)):
            return None
        return self.get_example(item)

    def __len__(self):
        if self.data is None:
            return 0
        return len(self.data)

    def get_example(self, i):
        if self.data is None:
            return None

        if self.label is None:
            return self.data_transform(self.data[i]), None
        else:
            return self.data_transform(self.data[i]), self.label_transform(self.label[i])

    def load_data(self):
        pass


class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))
        self.iter_count = None
        self.reset()

    def __iter__(self):
        return self

    def __next__(self):
        if self.iter_count >= len(self):
            self.reset()
            raise StopIteration

        i = self.iter_count * self.batch_size
        i_end = min(i + self.batch_size, len(self.dataset))
        batch = [self.dataset[j] for j in self.indexes[i:i_end]]

        data = np.array([example[0] for example in batch])
        label = np.array([example[1] for example in batch])

        self.iter_count += 1
        return data, label

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def reset(self):
        self.iter_count = 0
        if self.shuffle:
            np.random.shuffle(self.indexes)