from typing import (
    Callable,
    Iterator,
)

import tinytorch

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
    def __init__(self, get_params, lr, momentum: float = 0):
        super().__init__(get_params)
        self.lr = lr
        self.momentum = momentum
        self._init_momentum()
    
    def _init_momentum(self):
        if self.momentum != 0:
            self.momentum_buffer_list = []
            for param in self.get_params():
                self.momentum_buffer_list.append(
                    tinytorch.TensorBase.zeros_like(param.realize_cached_data())
                )

    def step(self):
        for i, param in enumerate(self.get_params()):
            param.realize_cached_data()
            if self.momentum != 0:
                self.momentum_buffer_list[i] = (
                    self.momentum_buffer_list[i] * self.momentum +
                    param.grad.realize_cached_data()
                )
                param.cached_data -= self.momentum_buffer_list[i] * self.lr
            else:
                param.cached_data -= param.grad.realize_cached_data() * self.lr
