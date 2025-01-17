from typing import (
    List,
    Optional,
    Tuple,
    Union,
)

import tinytorch
from tinytorch import TensorBase

class TensorOp:
    def __call__(self, *args):
        return Tensor.make_from_op(self, args)
    
    def compute(self, *args: Tuple[TensorBase]):
        raise NotImplementedError()
    
    def gradient(
        self, out_grad: "Tensor", node: "Tensor"
    ) -> Union["Tensor", Tuple["Tensor"]]:
        raise NotImplementedError()
    
    def gradient_as_tuple(
        self, out_grad: "Tensor", node: "Tensor"
    ) -> Tuple["Tensor"]:
        output = self.gradient(out_grad, node)
        if isinstance(output, tuple):
            return output
        elif isinstance(output, list):
            return tuple(output)
        else:
            return (output,)

class Tensor:
    op: Optional[TensorOp]
    inputs: List["Tensor"]
    requires_grad: bool = True
    cached_data: TensorBase
    grad: Optional["Tensor"]

    def realize_cached_data(self):
        if self.cached_data is not None:
            return self.cached_data
        self.cached_data = self.op.compute(
            *[x.realize_cached_data() for x in self.inputs]
        )
        return self.cached_data

    def is_leaf(self):
        return self.op is None
    
    def _init(
        self,
        op,
        inputs,
        *,
        num_outputs: int = 1,
        cached_data: List[object] = None,
        requires_grad: Optional[bool] = None
    ):
        if requires_grad is None:
            requires_grad = any(x.requires_grad for x in inputs)
        self.op = op
        self.inputs = inputs
        self.num_outputs = num_outputs
        self.cached_data = cached_data
        self.requires_grad = requires_grad

    def __init__(
        self,
        array,
        *,
        requires_grad=True,
    ):
        if isinstance(array, Tensor):
            cached_data = array.realize_cached_data()
        elif isinstance(array, TensorBase):
            cached_data = array
        else:
            raise NotImplementedError()
        self._init(
            None,
            [],
            cached_data=cached_data,
            requires_grad=requires_grad
        )
    
    @staticmethod
    def make_from_op(op: TensorOp, inputs: List["Tensor"]):
        tensor = Tensor.__new__(Tensor)
        tensor._init(op, inputs)
        if not tensor.requires_grad:
            return tensor.detach()
        tensor.realize_cached_data()
        return tensor

    @staticmethod
    def make_const(data, requires_grad=False):
        tensor = Tensor.__new__(Tensor)
        tensor._init(
            None,
            [],
            cached_data=data
            if not isinstance(data, Tensor)
            else data.realize_cached_data(),
            requires_grad=requires_grad
        )
        return tensor
    
    def detach(self):
        return Tensor.make_const(self.realize_cached_data())

    def backward(self, out_grad=None):
        from autodiff import compute_gradient_of_variables
        from utils import fill
        out_grad = (
            out_grad
            if out_grad
            else fill(1, self.realize_cached_data().shape, tinytorch.Device.get_default_device())
        )
        compute_gradient_of_variables(self, out_grad)
    
    def __add__(self, other):
        if isinstance(other, Tensor):
            return EWiseAdd()(self, other)
        else:
            return AddScalar(other)(self)
    
    def __mul__(self, other):
        if isinstance(other, Tensor):
            return EWiseMul()(self, other)
        else:
            return MulScalar(other)(self)
    
    def __sub__(self, other):
        if isinstance(other, Tensor):
            return EWiseAdd()(self, Negate()(other))
        else:
            return AddScalar(-other)(self)
    
    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return EWiseDiv()(self, other)
        else:
            return DivScalar(other)(self)
    
    def __neg__(self):
        return Negate()(self)
    
    def reshape(self, shape):
        return ReshapeOp(shape)(self)
    
    __radd__ = __add__
    __rmul__ = __mul__
    __rsub__ = __sub__

class EWiseAdd(TensorOp):
    def compute(self, a: TensorBase, b: TensorBase):
        return a + b
    
    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad

def add(a, b):
    return EWiseAdd()(a, b)

class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: TensorBase):
        return a + self.scalar
    
    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad

def add_scalar(a, scalar):
    return AddScalar(scalar)(a)

class EWiseMul(TensorOp):
    def compute(self, a: TensorBase, b: TensorBase):
        return a * b
    
    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs
    
def multiply(a, b):
    return EWiseMul()(a, b)

class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: TensorBase):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)

def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)

class EWiseDiv(TensorOp):
    def compute(self, a: TensorBase, b: TensorBase):
        return a / b

    def gradient(self, out_grad: Tensor, node: Tensor):
        a, b = node.inputs
        return out_grad / b, - node * out_grad / b

def divide(a, b):
    return EWiseDiv()(a, b)

class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: TensorBase):
        return a / self.scalar
        
    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad / self.scalar

def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)

class Negate(TensorOp):
    def compute(self, a: TensorBase):
        return -a
    
    def gradient(self, out_grad: Tensor, node: Tensor):
        return -out_grad

def negate(a):
    return Negate()(a)

class Log(TensorOp):
    def compute(self, a: TensorBase):
        return tinytorch.op.tensor_log(a)
    
    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad / node.inputs[0]

def log(a):
    return Log()(a)

class Exp(TensorOp):
    def compute(self, a: TensorBase):
        return tinytorch.op.tensor_exp(a)
    
    def gradient(self, out_grad: Tensor, node: Tensor):
        return node * out_grad
    
def exp(a):
    return Exp()(a)

class Conv2dOp(TensorOp):
    def __init__(self, stride, padding):
        self.stride = stride
        self.padding = padding

    def compute(self, input: TensorBase, weight: TensorBase, bias: TensorBase):
        return tinytorch.op.conv2d_forward(input, weight, bias, self.stride, self.padding)
    
    def gradient(self, out_grad: Tensor, node: Tensor):
        input, weight, bias = node.inputs
        grad_input, grad_weight, grad_bias = tinytorch.op.conv2d_backward(
            input.realize_cached_data(),
            weight.realize_cached_data(),
            out_grad.realize_cached_data(),
            self.stride,
            self.padding
        )
        return Tensor(grad_input), Tensor(grad_weight), Tensor(grad_bias)

def conv2d(input, weight, bias, stride, padding):
    return Conv2dOp(stride, padding)(input, weight, bias)

class CrossEntropyOp(TensorOp):
    def __init__(self, target):
        self.target = target.realize_cached_data()

    def compute(self, input: TensorBase):
        self.prob, loss = tinytorch.op.cross_entropy_forward(input, self.target)
        return loss
    
    def gradient(self, out_grad: Tensor, node: Tensor):
        grad_input = tinytorch.op.cross_entropy_backward(
            self.prob, self.target
        )
        grad_input *= out_grad.realize_cached_data().numpy()
        return Tensor(grad_input)

def cross_entropy(input, target):
    return CrossEntropyOp(target)(input)

class LinearOp(TensorOp):
    def compute(self, input: TensorBase, weight: TensorBase, bias: TensorBase):
        return tinytorch.op.linear_forward(input, weight, bias)

    def gradient(self, out_grad: Tensor, node: Tensor):
        input, weight, bias = node.inputs
        grad_input, grad_weight, grad_bias = tinytorch.op.linear_backward(
            input.realize_cached_data(),
            weight.realize_cached_data(),
            out_grad.realize_cached_data()
        )
        return Tensor(grad_input), Tensor(grad_weight), Tensor(grad_bias)

def linear(input, weight, bias):
    return LinearOp()(input, weight, bias)

class MaxPool2dOp(TensorOp):
    def __init__(self, kernel_size, stride, padding):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
    
    def compute(self, input: TensorBase):
        output, self.mask = tinytorch.op.max_pool2d_forward(
            input,
            self.kernel_size,
            self.stride,
            self.padding
        )
        return output
    
    def gradient(self, out_grad: Tensor, node: Tensor):
        grad_input = tinytorch.op.max_pool2d_backward(
            self.mask,
            out_grad.realize_cached_data(),
            self.kernel_size,
            self.stride,
            self.padding
        )
        return Tensor(grad_input)

def max_pool2d(input, kernel_size, stride, padding):
    return MaxPool2dOp(kernel_size, stride, padding)(input)

class ReLUOp(TensorOp):
    def compute(self, a: TensorBase):
        return tinytorch.op.relu_forward(a)
    
    def gradient(self, out_grad: Tensor, node: Tensor):
        a_grad = tinytorch.op.relu_backward(
            node.inputs[0].realize_cached_data(),
            out_grad.realize_cached_data()
        )
        return Tensor(a_grad)

def relu(a):
    return ReLUOp()(a)

class SigmoidOp(TensorOp):
    def compute(self, input: TensorBase):
        self.output =  tinytorch.op.sigmoid_forward(input)
        return self.output
    
    def gradient(self, out_grad: Tensor, node: Tensor):
        grad_input = tinytorch.op.sigmoid_backward(
            self.output,
            out_grad.realize_cached_data()
        )
        return Tensor(grad_input)

def sigmoid(a):
    return SigmoidOp()(a)

class ReshapeOp(TensorOp):
    def __init__(self, shape: List[int]):
        self.shape = shape

    def compute(self, a: TensorBase):
        return tinytorch.op.reshape(a, self.shape)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad.reshape(node.inputs[0].realize_cached_data().shape)

def reshape(a, shape):
    return ReshapeOp(shape)(a)

def flatten(input, start_dim: int = 0, end_dim: int = -1):
    shape = input.realize_cached_data().shape
    start_dim = start_dim if start_dim >= 0 else len(shape) + start_dim
    end_dim = end_dim if end_dim >= 0 else len(shape) + end_dim
    if start_dim < 0 or start_dim >= len(shape):
        raise Exception("start_dim out of range")
    if end_dim < 0 or end_dim >= len(shape):
        raise Exception("end_dim out of range")
    if start_dim > end_dim:
        raise Exception("start_dim should not be larger than end_dim")
    new_shape = []
    for i in range(len(shape)):
        if i <= start_dim or i > end_dim:
            new_shape.append(shape[i])
        else:
            new_shape[-1] *= shape[i]
    return reshape(input, new_shape)
