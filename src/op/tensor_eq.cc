#include "op/tensor_eq.h"

#include "utils/utils.h"
#include "tensor/tensor.cuh"
#include "backend/cuda/tensor_eq.cuh"

namespace TinyTorch {

bool tensor_eq(const Tensor &a, const Tensor &b) {
    ASSERT(a.device.is_cuda() && b.device.is_cuda());
    if (a.shape != b.shape) {
        return false;
    } else {
        return Backend::CUDA::tensor_eq(a.data, b.data, a.numel());
    }
}

}