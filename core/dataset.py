import numpy as np
import pinenut.core.cuda as cuda


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
        if not isinstance(item, (int, np.integer)):
            raise TypeError('dataset index must be an integer')
        if not 0 <= item < len(self):
            raise IndexError('dataset index out of range')
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
    def __init__(self, dataset, batch_size=1, shuffle=False, device='cpu',
                 drop_last=False):
        if isinstance(batch_size, bool) or not isinstance(
                batch_size, (int, np.integer)):
            raise TypeError('batch_size must be an integer')
        if batch_size <= 0:
            raise ValueError('batch_size must be greater than zero')
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))
        self.iter_count = None
        self.device = 'cpu'
        self.device_list = None
        self.drop_last = drop_last
        self.reset()
        if device != 'cpu':
            self.cuda(device)

    def __iter__(self):
        return self

    def __next__(self):
        if self.iter_count >= len(self):
            self.reset()
            raise StopIteration

        i = self.iter_count * self.batch_size
        i_end = min(i + self.batch_size, len(self.dataset))
        batch = [self.dataset[j] for j in self.indexes[i:i_end]]

        # do not use gpu
        if self.device == 'cpu':
            data = np.array([example[0] for example in batch])
            label = np.array([example[1] for example in batch])
            self.iter_count += 1
            return data, label

        # try to use gpu
        gpu_count = 1
        if self.device_list is not None:
            gpu_count = len(self.device_list)

        if gpu_count <= 1:
            cp = cuda.cupy()
            with cp.cuda.Device(cuda._device_index(self.device)):
                data = cp.array([example[0] for example in batch])
                label = cp.array([example[1] for example in batch])
        else:
            # if count of gpu more then 1, should divide data into count in order to process  parallel by multi threads
            m = self.batch_size // gpu_count  # notice: batch_size should be divisible by gpu_count
            assert m * gpu_count == self.batch_size
            cp = cuda.cupy()

            data = []
            label = []
            for i in range(gpu_count):
                gpu_id = self.device_list[i]
                with cp.cuda.Device(gpu_id):
                    mini_batch = batch[i * m: (i + 1) * m]
                    mini_data = cp.array([example[0] for example in mini_batch])
                    mini_label = cp.array([example[1] for example in mini_batch])
                    data.append(mini_data)
                    label.append(mini_label)

        self.iter_count += 1
        return data, label

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def reset(self):
        self.iter_count = 0
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def cuda(self, device=None, device_list=None):
        if not cuda.is_available():
            raise RuntimeError('CUDA is not available')
        self.device = 'cuda' if device is None else device
        self.device_list = device_list
        return self

    def cpu(self):
        self.device = 'cpu'
        self.device_list = None
        return self

    def to(self, device, device_list=None):
        if device == 'cpu':
            return self.cpu()
        return self.cuda(device=device, device_list=device_list)
