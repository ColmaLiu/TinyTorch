import tinytorch

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk
from torchvision import transforms
import os

def transform(example):
    image = example["image"]
    if len(image.split()) != 3:
        image = image.convert("RGB")
    image = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])(image)
    example["image"] = image
    return example

def collate_fn(data):
    images, labels = [], []
    for example in data:
        image, label = example["image"], example["label"]
        if len(image.split()) != 3:
            image = image.convert("RGB")
        images.append(transforms.Compose([transforms.ToTensor()])(image))
        labels.append(label)
    return (torch.stack(tuple(images)), torch.tensor(labels))

preprocessed_path = "./data/tiny-imagenet-preprocessed/"
if os.path.exists(preprocessed_path):
    dataset = load_from_disk(preprocessed_path)
else:
    path = "./data/tiny-imagenet/"
    if not os.path.exists(path):
        download_path = "zh-plus/tiny-imagenet"
        dataset = load_dataset(download_path)
        dataset.save_to_disk(path)
    dataset = load_from_disk(path)
    dataset = dataset.map(transform)
    dataset.save_to_disk(preprocessed_path)
dataset = dataset.with_format("torch")

trainloader = DataLoader(dataset["train"], batch_size=64, shuffle=True, num_workers=16)
testloader = DataLoader(dataset["valid"], batch_size=1000, shuffle=False, num_workers=16)

def toTensor(x: torch.Tensor):
    return tinytorch.TensorBase(x.float().detach().numpy())

def main():
    # Example for usage
    # for data in trainloader:
    #     image, label = data.values()
    pass

if __name__ == "__main__":
    main()
