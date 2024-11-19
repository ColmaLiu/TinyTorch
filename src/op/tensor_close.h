#ifndef TINYTORCH_OP_TENSOR_CLOSE_H_
#define TINYTORCH_OP_TENSOR_CLOSE_H_

#include "tensor/tensor.cuh"

namespace TinyTorch {

bool tensor_close(const Tensor &a, const Tensor &b);

}

#endif  // TINYTORCH_OP_TENSOR_CLOSE_H_