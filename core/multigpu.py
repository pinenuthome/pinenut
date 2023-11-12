import numpy as np
import pinenut.core.cuda as cuda
import pinenut.core.dataset as ds

from pinenut.core import (DataLoader, relu, accuracy, MLP, softmax, softmax_cross_entropy, Adam)


class ParallelModel:
    def __init__(self, device_list, layer_output_sizes=(), hidden_activation=None, output_activation=None):
        gpu_count = len(device_list)
        if gpu_count <= 1:
            raise "gpu count must be more than ONE when using class ParallelModel."
        self.gpu_count = gpu_count
        self.device_list = device_list
        self.layer_output_sizes = layer_output_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

    def train_gpus(self, train_dataset, epochs, batch_size):
        train_loader = ds.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        gpu_count = self.gpu_count
        device_list = self.device_list
        layer_output_sizes =  self.layer_output_sizes
        cp = cuda.Cuda.cupy()

        gpu_0 = device_list[0]
        models = {}
        for i in range(gpu_count):
            gpu_id = device_list[i]
            with cp.cuda.Device(gpu_id):
                new_model = MLP(self.layer_output_sizes, hidden_activation=self.hidden_activation, output_activation=self.output_activation)
                new_model.to_gpu()
                models[gpu_id] = new_model

        optimizer = Adam(models[gpu_0])
        train_loader.to_gpu(device_list=device_list)

        for epoch in range(epochs):
            sum_loss = 0.0
            sum_acc = 0.0
            for x, y in train_loader:
                # forward
                y_pred = {}
                loss = {}
                for i in range(gpu_count):
                    gpu_id = device_list[i]
                    with cp.cuda.Device(gpu_id):
                        y_pred[gpu_id] = models[gpu_id](x[i])  # do forward
                        loss[gpu_id] = softmax_cross_entropy(y_pred[gpu_id], y[i])

                # backward
                for i in range(gpu_count):
                    gpu_id = device_list[i]
                    with cp.cuda.Device(gpu_id):
                        # run backward
                        loss[gpu_id].backward(enable_buildgraph=False)

                # gather all grad to gpu_0
                for j in range(1, gpu_count):
                    gpu_id = device_list[i]
                    for a, b in zip(models[0].params, models[j].params):
                        with cp.cuda.Device(gpu_0):
                            b0 = cp.array(b.grad.data)  # grad
                            a.grad.data += b0
                            # a.grad.data /= 2

                # update params on gpu_0
                with cp.cuda.Device(gpu_0):
                    optimizer.update()  # gpu_0 update params

                # copy new params from gpu_0 to others
                for j in range(1, gpu_count):
                    gpu_id = device_list[i]
                    for a, b in zip(models[0].params, models[j].params):
                        with cp.cuda.Device(gpu_id):
                            a0 = cp.array(a.data)  # copy data from gpu_0 to gpu_x
                        b.data = a0  # debug to check whether change original value
                
                # clear grad
                for i in range(0, gpu_count):
                    gpu_id = device_list[i]
                    models[gpu_id].clear_all_grad()
                    loss[gpu_id].unchain_backward()  # release memory    

                acc = accuracy(y_pred[0], y[0])
                sum_loss += float(loss[0].data) * len(y[0])
                sum_acc += float(acc.data) * len(y[0])

            train_loader.reset()
            print('epoch:{} '.format(epoch + 1))
            tl = sum_loss / (len(train_dataset) / gpu_count)
            print('train loss: {:.4f}, accuracy:{:.4f}'
                  .format(sum_loss / (len(train_dataset) / gpu_count), sum_acc / (len(train_dataset) / gpu_count)))
