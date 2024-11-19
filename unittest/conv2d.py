import tinytorch

import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

BATCHSIZE = 64
CHANNELS_IN = 3
CHANNELS_OUT = 6
HEIGHT = 28
WIDTH = 28
KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1

HEIGHT_OUT = (HEIGHT + 2 * PADDING - KERNEL_SIZE) // STRIDE + 1
WIDTH_OUT = (WIDTH + 2 * PADDING - KERNEL_SIZE) // STRIDE + 1

device = None

def test_conv2d():
    input = torch.randn((BATCHSIZE, CHANNELS_IN, HEIGHT, WIDTH), device=device, requires_grad=True)
    weight = torch.randn((CHANNELS_OUT, CHANNELS_IN, KERNEL_SIZE, KERNEL_SIZE), device=device, requires_grad=True)
    bias = torch.randn((CHANNELS_OUT, ), device=device, requires_grad=True)
    grad_output = torch.randn((BATCHSIZE, CHANNELS_OUT, HEIGHT_OUT, WIDTH_OUT), device=device, requires_grad=True)
    output_gt = F.conv2d(input, weight, bias, STRIDE, PADDING)
    output_gt.backward(grad_output)

    output_gt = tinytorch.Tensor(output_gt.detach().numpy())
    grad_input_gt = tinytorch.Tensor(input.grad.detach().numpy())
    grad_weight_gt = tinytorch.Tensor(weight.grad.detach().numpy())
    grad_bias_gt = tinytorch.Tensor(bias.grad.detach().numpy())
    input = tinytorch.Tensor(input.detach().numpy())
    weight = tinytorch.Tensor(weight.detach().numpy())
    bias = tinytorch.Tensor(bias.detach().numpy())
    grad_output = tinytorch.Tensor(grad_output.detach().numpy())

    output_res = tinytorch.op.conv2d_forward(input, weight, bias, STRIDE, PADDING)
    grad_input_res, grad_weight_res, grad_bias_res = tinytorch.op.conv2d_backward(input, weight, grad_output, STRIDE, PADDING)

    # if errors accumulate and assert fails, `assert tinytorch.op.tensor_close(res, gt)` may be useful,
    # which is a looser test
    assert output_res == output_gt
    assert grad_input_res == grad_input_gt
    assert grad_weight_res == grad_weight_gt
    assert grad_bias_res == grad_bias_gt

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Torch Device:", device)
    tinytorch.Device.set_default_device(tinytorch.Device.cuda())
    print("TinyTorch Default Device:", "cpu" if tinytorch.Device.get_default_device().is_cpu() else "cuda")
    test_conv2d()

if __name__ == "__main__":
    main()
