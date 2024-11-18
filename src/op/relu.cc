#include "op/relu.h"

#include "utils/utils.h"
#include "tensor/tensor.cuh"
#include "backend/cuda/relu.cuh"

namespace TinyTorch {

Tensor relu_forward(const Tensor &input) {
    ASSERT(input.device.is_cuda());
    Tensor output(input.shape, input.device);
    if (input.device.is_cuda()) {
        Backend::CUDA::relu_forward(input.data, output.data, input.numel());
    } else {}
    return output;
}

// return: grad_input
Tensor relu_backward(const Tensor &input, const Tensor &grad_output) {
    ASSERT(input.device.is_cuda() && grad_output.device.is_cuda());
    ASSERT(input.shape == grad_output.shape);
    Tensor grad_input(input.shape, input.device);
    if (input.device.is_cuda()) {
        Backend::CUDA::relu_backward(grad_output.data, input.data, grad_input.data, input.numel());
    } else {}
    return grad_input;
}

}