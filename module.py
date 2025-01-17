from typing import (
    Any,
    Callable,
    Iterator,
)
from math import sqrt

from basic import *
from utils import rand

class Module:
    def parameters(self) -> Iterator[Tensor]:
        attrs = vars(self)
        for attr_name, attr_value in attrs.items():
            if isinstance(attr_value, Tensor):
                yield attr_value
            elif isinstance(attr_value, Module):
                yield from attr_value.parameters()
    
    def forward(self, *args):
        raise NotImplementedError
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class Conv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0
    ) -> None:
        self.stride = stride
        self.padding = padding
        self.weight = (rand(
            [out_channels, in_channels, kernel_size, kernel_size],
            tinytorch.Device.get_default_device(),
            requires_grad = True
        ) - 0.5) * (2 / (sqrt(in_channels) * kernel_size))
        self.bias = (rand(
            [out_channels],
            tinytorch.Device.get_default_device(),
            requires_grad = True
        ) - 0.5) * (2 / (sqrt(in_channels) * kernel_size))

    def forward(self, input: Tensor) -> Tensor:
        return conv2d(input, self.weight, self.bias, self.stride, self.padding)

class CrossEntropy(Module):
    def __init__(self) -> None:
        pass

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return cross_entropy(input, target)

class Linear(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
    ) -> None:
        self.weight = (rand(
            [in_features, out_features],
            tinytorch.Device.get_default_device(),
            requires_grad = True
        ) - 0.5) * (2 / sqrt(in_features))
        self.bias = (rand(
            [out_features],
            tinytorch.Device.get_default_device(),
            requires_grad = True
        ) - 0.5) * (2 / sqrt(in_features))

    def forward(self, input: Tensor) -> Tensor:
        return linear(input, self.weight, self.bias)

class MaxPool2d(Module):
    def __init__(
        self,
        kernel_size: int,
        padding: int = 0,
    ) -> None:
        self.kernel_size = kernel_size
        self.stride = kernel_size
        self.padding = padding

    def forward(self, input: Tensor) -> Tensor:
        return max_pool2d(input, self.kernel_size, self.stride, self.padding)

class ReLU(Module):
    def __init__(self) -> None:
        pass

    def forward(self, input: Tensor) -> Tensor:
        return relu(input)

class Sigmoid(Module):
    def __init__(self) -> None:
        pass

    def forward(self, input: Tensor) -> Tensor:
        return sigmoid(input)