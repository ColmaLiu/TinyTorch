from typing import (
    Callable,
    Iterator,
)

from basic import Tensor

class Optimizer:
    def __init__(self, get_params: Callable[[], Iterator[Tensor]]):
        self.get_params = get_params
        self.zero_grad()
    
    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for param in self.get_params():
            param.op = None
            param.grad = None
            param.inputs = []

class SGD(Optimizer):
    def __init__(self, get_params, lr):
        super().__init__(get_params)
        self.lr = lr

    def step(self):
        for param in self.get_params():
            param.realize_cached_data()
            param.cached_data -= param.grad.realize_cached_data() * self.lr

# from module import Module
# import tinytorch

# tinytorch.Device.set_default_device(tinytorch.Device.cuda())
# model = Module()
# optimizer = SGD(model.parameters, 0.1)
# print(model.x.op)
# # print(model.x.grad)
# optimizer.zero_grad()
# print(model.x.op)