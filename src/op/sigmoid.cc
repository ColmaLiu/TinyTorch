#include "op/sigmoid.h"

#include "utils/utils.h"
#include "tensor/tensor.cuh"
#include "backend/cuda/sigmoid.cuh"

namespace TinyTorch {

Tensor sigmoid_forward(const Tensor &input) {
    ASSERT(input.device.is_cuda());
    Tensor output(input.shape, input.device);
    if (input.device.is_cuda()) {
        Backend::CUDA::sigmoid_forward(input.data, output.data, input.numel());
    } else {}
    return output;
}

Tensor sigmoid_backward(const Tensor &output, const Tensor &grad_output) {
    ASSERT(output.device.is_cuda() && grad_output.device.is_cuda());
    ASSERT(output.shape == grad_output.shape);
    Tensor grad_input(output.shape, output.device);
    if (output.device.is_cuda()) {
        Backend::CUDA::sigmoid_backward(grad_output.data, output.data, grad_input.data, output.numel());
    } else {}
    return grad_input;
}

}