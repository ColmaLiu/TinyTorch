#include "op/tensor_binary_op.h"

#include "utils/utils.h"
#include "tensor/tensor.cuh"
#include "backend/cuda/tensor_binary_op.cuh"

namespace TinyTorch {

Tensor tensor_add(const Tensor &a, const Tensor &b) {
    ASSERT(a.device.is_cuda() && b.device.is_cuda());
    ASSERT(a.shape == b.shape);
    Tensor res(a.shape, a.device);
    if (a.device.is_cuda()) {
        Backend::CUDA::tensor_add(a.data, b.data, res.data, a.numel());
    } else {}
    return res;
}

Tensor tensor_sub(const Tensor &a, const Tensor &b) {
    ASSERT(a.device.is_cuda() && b.device.is_cuda());
    ASSERT(a.shape == b.shape);
    Tensor res(a.shape, a.device);
    if (a.device.is_cuda()) {
        Backend::CUDA::tensor_sub(a.data, b.data, res.data, a.numel());
    } else {}
    return res;
}

Tensor tensor_mul(const Tensor &a, const Tensor &b) {
    ASSERT(a.device.is_cuda() && b.device.is_cuda());
    ASSERT(a.shape == b.shape);
    Tensor res(a.shape, a.device);
    if (a.device.is_cuda()) {
        Backend::CUDA::tensor_mul(a.data, b.data, res.data, a.numel());
    } else {}
    return res;
}

Tensor tensor_div(const Tensor &a, const Tensor &b) {
    ASSERT(a.device.is_cuda() && b.device.is_cuda());
    ASSERT(a.shape == b.shape);
    Tensor res(a.shape, a.device);
    if (a.device.is_cuda()) {
        Backend::CUDA::tensor_div(a.data, b.data, res.data, a.numel());
    } else {}
    return res;
}

}