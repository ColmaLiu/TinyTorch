import tinytorch

from cnn_tiny_imagenet.dataset import trainloader, testloader, toTensor

from module import *
from optimizer import *

import torch

class Model(Module):
    def __init__(self):
        self.pool1 = MaxPool2d(2)
        self.pool2 = MaxPool2d(3)
        self.pool3 = MaxPool2d(2)
        self.conv1 = Conv2d(3, 16, 5, padding=2)
        self.conv2 = Conv2d(16, 16, 3, padding=0)
        self.conv3 = Conv2d(16, 32, 3, padding=0)
        self.fc1 = Linear(32 * 4 * 4, 200)

    def forward(self, x):
        x = self.pool1(relu(self.conv1(x)))
        x = self.pool2(relu(self.conv2(x)))
        x = self.pool3(relu(self.conv3(x)))
        x = flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        return x

def main():
    tinytorch.Device.set_default_device(tinytorch.Device.cuda())
    print("TinyTorch Default Device:", "cpu" if tinytorch.Device.get_default_device().is_cpu() else "cuda")
    model = Model()
    criterion = CrossEntropy()
    optimizer = SGD(model.parameters, lr=0.01, momentum=0.9)

    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs_pt, labels_pt = data.values()
            inputs, labels = Tensor(toTensor(inputs_pt)), Tensor(toTensor(labels_pt))

            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.realize_cached_data().numpy()
            if i % 100 == 99:    # print every 100 mini-batches
                minibatches_loss = running_loss / 100
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {minibatches_loss:.3f}', flush=True)
                running_loss = 0.0

            # print(f'[{epoch + 1}, {i:5d}] loss: {loss.realize_cached_data().numpy():.3f}', flush=True)
        
        test_loss = 0
        correct = 0
        batches = 0
        for data in testloader:
            inputs_pt, labels_pt = data.values()
            inputs, labels = Tensor(toTensor(inputs_pt)), Tensor(toTensor(labels_pt))

            output = model(inputs)
            test_loss += criterion(output, labels).realize_cached_data().numpy()
            # Nothing is related to the training process and auto differentiation here,
            # and it is just some data processing,
            # so I use some PyTorch functions to handle it.
            output = torch.tensor(output.realize_cached_data().numpy())
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels_pt.data.view_as(pred)).sum()
            batches += 1
        test_loss /= batches
        print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(testloader.dataset)} ({100. * correct / len(testloader.dataset):.2f}%)\n')

if __name__ == "__main__":
    main()
