#ifndef TINYTORCH_OP_TENSOR_BINARY_OP_H_
#define TINYTORCH_OP_TENSOR_BINARY_OP_H_

#include "tensor/tensor.cuh"

namespace TinyTorch {

Tensor tensor_add(const Tensor &a, const Tensor &b);

Tensor tensor_sub(const Tensor &a, const Tensor &b);

Tensor tensor_mul(const Tensor &a, const Tensor &b);

Tensor tensor_div(const Tensor &a, const Tensor &b);

}

#endif  // TINYTORCH_OP_TENSOR_BINARY_OP_H_