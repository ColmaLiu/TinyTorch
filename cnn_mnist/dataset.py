import tinytorch

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = datasets.MNIST('./data/', download=True, train=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)

testset = datasets.MNIST('./data/', download=True, train=False, transform=transform)
testloader = DataLoader(testset, batch_size=1000, shuffle=False, num_workers=4)

def toTensor(x: torch.Tensor):
    return tinytorch.TensorBase(x.float().detach().numpy())

def main():
    print(trainset[0])
    return
    # Example with torch.utils.data.Dataloader
    for inputs, labels in trainloader:
        inputs, labels = toTensor(inputs), toTensor(labels)

    # Example without torch.utils.data.Dataloader
    # batchsize = 64
    # train = []
    # for _ in range((len(trainset) - 1) // batchsize + 1):
    #     imgs, targets = [], []
    #     for i in range(batchsize):
    #         if _ * batchsize + i < len(trainset):
    #             img, target = trainset[_ * batchsize + i]
    #             imgs.append(img)
    #             targets.append(target)
    #     imgs = torch.stack(tuple(imgs))
    #     targets = torch.Tensor(targets)
    #     train.append((imgs, targets))
    # for i in range(len(train)):
    #     imgs, targets = train[i]
    #     train[i] = (toTensor(imgs), toTensor(targets))
    #     # print(type(train[i][0]), train[i][0].shape)
    #     # print(type(train[i][1]), train[i][1].shape)
    # pass

if __name__ == "__main__":
    main()
