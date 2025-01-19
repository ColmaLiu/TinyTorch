import tinytorch

import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

SHAPE = [32, 3, 28, 28]

device = None

def test_sigmoid():
    input = torch.randn(SHAPE, device=device, requires_grad=True)
    grad_output = torch.randn(SHAPE, device=device, requires_grad=True)
    output_gt = F.sigmoid(input)
    output_gt.backward(grad_output)

    output_gt = tinytorch.TensorBase(output_gt.detach().numpy())
    grad_input_gt = tinytorch.TensorBase(input.grad.detach().numpy())
    input = tinytorch.TensorBase(input.detach().numpy())
    grad_output = tinytorch.TensorBase(grad_output.detach().numpy())

    output_res = tinytorch.op.sigmoid_forward(input)
    grad_input_res = tinytorch.op.sigmoid_backward(output_res, grad_output)

    assert output_res == output_gt
    assert grad_input_res == grad_input_gt

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Torch Device:", device)
    tinytorch.Device.set_default_device(tinytorch.Device.cuda())
    print("TinyTorch Default Device:", "cpu" if tinytorch.Device.get_default_device().is_cpu() else "cuda")
    test_sigmoid()

if __name__ == "__main__":
    main()
