import tinytorch

import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

BATCHSIZE = 32
CHANNELS = 3
HEIGHT = 28
WIDTH = 28
# stride should be equal to ksize
KERNEL_SIZE = 2
STRIDE = 2
PADDING = 0

HEIGHT_OUT = (HEIGHT + 2 * PADDING - KERNEL_SIZE) // STRIDE + 1
WIDTH_OUT = (WIDTH + 2 * PADDING - KERNEL_SIZE) // STRIDE + 1

device = None

def test_max_pool2d():
    input = torch.randn((BATCHSIZE, CHANNELS, HEIGHT, WIDTH), device=device, requires_grad=True)
    grad_output = torch.randn((BATCHSIZE, CHANNELS, HEIGHT_OUT, WIDTH_OUT), device=device, requires_grad=True)
    output_gt = F.max_pool2d(input, KERNEL_SIZE, STRIDE, PADDING)
    output_gt.backward(grad_output)

    output_gt = tinytorch.Tensor(output_gt.detach().numpy())
    grad_input_gt = tinytorch.Tensor(input.grad.detach().numpy())
    input = tinytorch.Tensor(input.detach().numpy())
    grad_output = tinytorch.Tensor(grad_output.detach().numpy())

    output_res, mask = tinytorch.op.max_pool2d_forward(input, KERNEL_SIZE, STRIDE, PADDING)
    grad_input_res = tinytorch.op.max_pool2d_backward(mask, grad_output, KERNEL_SIZE, STRIDE, PADDING)

    assert output_res == output_gt
    assert grad_input_res == grad_input_gt

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Torch Device:", device)
    tinytorch.Device.set_default_device(tinytorch.Device.cuda())
    print("TinyTorch Default Device:", "cpu" if tinytorch.Device.get_default_device().is_cpu() else "cuda")
    test_max_pool2d()

if __name__ == "__main__":
    main()
