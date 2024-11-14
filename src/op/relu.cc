#include "op/relu.h"

#include <cassert>

#include "tensor/tensor.cuh"
#include "backend/cuda/relu.cuh"

namespace TinyTorch {

Tensor relu_forward(const Tensor &input) {
    assert(input.device.is_cuda());
    Tensor output(input.shape, input.device);
    if (input.device.is_cuda()) {
        Backend::CUDA::relu_forward(input.data, output.data, input.numel());
    } else {}
    return output;
}

// return: grad_input
Tensor relu_backward(const Tensor &input, const Tensor &grad_output) {
    assert(input.device.is_cuda() && grad_output.device.is_cuda());
    assert(input.shape == grad_output.shape);
    Tensor grad_input(input.shape, input.device);
    if (input.device.is_cuda()) {
        Backend::CUDA::relu_backward(grad_output.data, input.data, grad_input.data, input.numel());
    } else {}
    return grad_input;
}

}