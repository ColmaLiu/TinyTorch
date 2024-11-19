#include "op/tensor_close.h"

#include "utils/utils.h"
#include "tensor/tensor.cuh"
#include "backend/cuda/tensor_close.cuh"

namespace TinyTorch {

bool tensor_close(const Tensor &a, const Tensor &b) {
    ASSERT(a.device.is_cuda() && b.device.is_cuda());
    if (a.shape != b.shape) {
        return false;
    } else {
        return Backend::CUDA::tensor_close(a.data, b.data, a.numel());
    }
}

}