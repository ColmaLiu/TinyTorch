#include "op/tensor_scalar_op.h"

#include "utils/utils.h"
#include "tensor/tensor.cuh"
#include "backend/cuda/tensor_scalar_op.cuh"

namespace TinyTorch {

Tensor tensor_adds(const Tensor &input, const float scalar) {
    ASSERT(input.device.is_cuda());
    Tensor output(input.shape, input.device);
    if (input.device.is_cuda()) {
        Backend::CUDA::tensor_adds(input.data, output.data, scalar, input.numel());
    } else {}
    return output;
}

Tensor tensor_subs(const Tensor &input, const float scalar) {
    ASSERT(input.device.is_cuda());
    Tensor output(input.shape, input.device);
    if (input.device.is_cuda()) {
        Backend::CUDA::tensor_subs(input.data, output.data, scalar, input.numel());
    } else {}
    return output;
}

Tensor tensor_muls(const Tensor &input, const float scalar) {
    ASSERT(input.device.is_cuda());
    Tensor output(input.shape, input.device);
    if (input.device.is_cuda()) {
        Backend::CUDA::tensor_muls(input.data, output.data, scalar, input.numel());
    } else {}
    return output;
}

Tensor tensor_divs(const Tensor &input, const float scalar) {
    ASSERT(input.device.is_cuda());
    Tensor output(input.shape, input.device);
    if (input.device.is_cuda()) {
        Backend::CUDA::tensor_divs(input.data, output.data, scalar, input.numel());
    } else {}
    return output;
}

Tensor tensor_pows(const Tensor &input, const float scalar) {
    ASSERT(input.device.is_cuda());
    Tensor output(input.shape, input.device);
    if (input.device.is_cuda()) {
        Backend::CUDA::tensor_pows(input.data, output.data, scalar, input.numel());
    } else {}
    return output;
}

}