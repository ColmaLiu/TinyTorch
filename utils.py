from basic import Tensor

import tinytorch

def zeros(shape, device, requires_grad=False):
    array = tinytorch.TensorBase.zeros(shape, device)
    return Tensor(array, requires_grad=requires_grad)

def zeros_like(other: Tensor, requires_grad=False):
    array = tinytorch.TensorBase.zeros_like(other.realize_cached_data())
    return Tensor(array, requires_grad=requires_grad)

def fill(scalar, shape, device, requires_grad=False):
    array = tinytorch.TensorBase.fill(scalar, shape, device)
    return Tensor(array, requires_grad=requires_grad)

def fill_like(other: Tensor, requires_grad=False):
    array = tinytorch.TensorBase.fill_like(other.realize_cached_data())
    return Tensor(array, requires_grad=requires_grad)

def rand(shape, device, requires_grad=False):
    array = tinytorch.TensorBase.rand(shape, device)
    return Tensor(array, requires_grad=requires_grad)

def rand_like(other: Tensor, requires_grad=False):
    array = tinytorch.TensorBase.rand_like(other.realize_cached_data())
    return Tensor(array, requires_grad=requires_grad)

def randn(shape, device, requires_grad=False):
    array = tinytorch.TensorBase.randn(shape, device)
    return Tensor(array, requires_grad=requires_grad)

def randn_like(other: Tensor, requires_grad=False):
    array = tinytorch.TensorBase.randn_like(other.realize_cached_data())
    return Tensor(array, requires_grad=requires_grad)