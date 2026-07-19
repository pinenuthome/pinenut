import numpy as np

import pinenut as pn
from pinenut import datasets, nn
from pinenut.nn import functional as F
from pinenut.utils.data import DataLoader


def data_transform(value):
    return value.flatten().astype(np.float32) / 255.0


test_dataset = datasets.MNIST(train=False, data_transform=data_transform)
model = nn.MLP(784, (100, 10), hidden_activation=F.relu)
model.load_state_dict(pn.load('mnist_weights.npz'))
model.eval()

device = 'cuda' if pn.cuda.is_available() else 'cpu'
model.to(device)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
test_loader.to(device)

total_loss = 0.0
total_accuracy = 0.0
with pn.no_grad():
    for inputs, labels in test_loader:
        predictions = model(inputs)
        loss = F.cross_entropy(predictions, labels)
        batch_accuracy = F.accuracy(predictions, labels)
        total_loss += float(pn.cuda.as_numpy(loss.data)) * len(labels)
        total_accuracy += float(
            pn.cuda.as_numpy(batch_accuracy.data)) * len(labels)

print('test loss: {:.4f}, accuracy:{:.4f}'.format(
    total_loss / len(test_dataset),
    total_accuracy / len(test_dataset),
))
