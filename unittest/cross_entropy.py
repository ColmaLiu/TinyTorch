import tinytorch

import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

BATCHSIZE = 32
LABELS = 10

device = None

def test_cross_entropy():
    input = torch.randn((BATCHSIZE, LABELS), device=device, requires_grad=True)
    target = torch.randint(0, LABELS, (BATCHSIZE, ), device=device)
    loss_gt = F.cross_entropy(input, target)
    loss_gt.backward()

    grad_input_gt = tinytorch.TensorBase(input.grad.detach().numpy())
    input = tinytorch.TensorBase(input.detach().numpy())
    target = tinytorch.TensorBase(target.float().detach().numpy())
    loss_gt = tinytorch.TensorBase(loss_gt.item())

    prob_res, loss_res = tinytorch.op.cross_entropy_forward(input, target)
    grad_input_res = tinytorch.op.cross_entropy_backward(prob_res, target)

    assert loss_res == loss_gt
    assert grad_input_res == grad_input_gt

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Torch Device:", device)
    tinytorch.Device.set_default_device(tinytorch.Device.cuda())
    print("TinyTorch Default Device:", "cpu" if tinytorch.Device.get_default_device().is_cpu() else "cuda")
    test_cross_entropy()

if __name__ == "__main__":
    main()
