import tinytorch

import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

BATCHSIZE = 10
IN_FEATURES = 32
OUT_FEATURES = 64

device = None

def test_linear():
    input = torch.randn((BATCHSIZE, IN_FEATURES), device=device, requires_grad=True)
    weight = torch.randn((OUT_FEATURES, IN_FEATURES), device=device, requires_grad=True)
    bias = torch.randn(OUT_FEATURES, device=device, requires_grad=True)
    grad_output = torch.randn((BATCHSIZE, OUT_FEATURES), device=device, requires_grad=True)
    output_gt = F.linear(input, weight, bias)
    output_gt.backward(grad_output)

    output_gt = tinytorch.TensorBase(output_gt.detach().numpy())
    grad_input_gt = tinytorch.TensorBase(input.grad.detach().numpy())
    grad_weight_gt = tinytorch.TensorBase(weight.grad.transpose(0, 1).detach().numpy())
    grad_bias_gt = tinytorch.TensorBase(bias.grad.detach().numpy())
    input = tinytorch.TensorBase(input.detach().numpy())
    weight = tinytorch.TensorBase(weight.transpose(0, 1).detach().numpy())
    bias = tinytorch.TensorBase(bias.detach().numpy())
    grad_output = tinytorch.TensorBase(grad_output.detach().numpy())

    output_res = tinytorch.op.linear_forward(input, weight, bias)
    grad_input_res, grad_weight_res, grad_bias_res = tinytorch.op.linear_backward(input, weight, grad_output)

    assert output_res == output_gt
    assert grad_input_res == grad_input_gt
    assert grad_weight_res == grad_weight_gt
    assert grad_bias_res == grad_bias_gt

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Torch Device:", device)
    tinytorch.Device.set_default_device(tinytorch.Device.cuda())
    print("TinyTorch Default Device:", "cpu" if tinytorch.Device.get_default_device().is_cpu() else "cuda")
    test_linear()

if __name__ == "__main__":
    main()
