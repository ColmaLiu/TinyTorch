#include "op/tensor_unary_op.h"

#include "utils/utils.h"
#include "tensor/tensor.cuh"
#include "backend/cuda/tensor_unary_op.cuh"

namespace TinyTorch {

Tensor tensor_neg(const Tensor &input) {
    ASSERT(input.device.is_cuda());
    Tensor output(input.shape, input.device);
    if (input.device.is_cuda()) {
        Backend::CUDA::tensor_neg(input.data, output.data, input.numel());
    } else {}
    return output;
}

Tensor tensor_inv(const Tensor &input) {
    ASSERT(input.device.is_cuda());
    Tensor output(input.shape, input.device);
    if (input.device.is_cuda()) {
        Backend::CUDA::tensor_inv(input.data, output.data, input.numel());
    } else {}
    return output;
}

Tensor tensor_exp(const Tensor &input) {
    ASSERT(input.device.is_cuda());
    Tensor output(input.shape, input.device);
    if (input.device.is_cuda()) {
        Backend::CUDA::tensor_exp(input.data, output.data, input.numel());
    } else {}
    return output;
}

Tensor tensor_log(const Tensor &input) {
    ASSERT(input.device.is_cuda());
    Tensor output(input.shape, input.device);
    if (input.device.is_cuda()) {
        Backend::CUDA::tensor_log(input.data, output.data, input.numel());
    } else {}
    return output;
}

}