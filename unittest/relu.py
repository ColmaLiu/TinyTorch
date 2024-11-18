import tinytorch

import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

SHAPE = [3200, 3, 28, 28]

def test_relu():
    input = torch.randn(SHAPE, requires_grad=True)
    grad_output = torch.randn(SHAPE, requires_grad=True)
    output_gt = F.relu(input)
    output_gt.backward(grad_output)

    output_gt = output_gt.detach().numpy()
    grad_input_gt = input.grad.detach().numpy()
    input = tinytorch.Tensor(input.detach().numpy())
    grad_output = tinytorch.Tensor(grad_output.detach().numpy())

    output_res = tinytorch.op.relu_forward(input)
    grad_input_res = tinytorch.op.relu_backward(input, grad_output)

    np.testing.assert_allclose(output_res.numpy(), output_gt)
    np.testing.assert_allclose(grad_input_res.numpy(), grad_input_gt)

def main():
    tinytorch.Device.set_default_device(tinytorch.Device.cuda())
    print("Device:", "cpu" if tinytorch.Device.get_default_device().is_cpu() else "cuda")
    test_relu()

if __name__ == "__main__":
    main()
