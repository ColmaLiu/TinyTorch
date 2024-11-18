#ifndef TINYTORCH_OP_TENSOR_EQ_H_
#define TINYTORCH_OP_TENSOR_EQ_H_

#include "tensor/tensor.cuh"

namespace TinyTorch {

bool tensor_eq(const Tensor &a, const Tensor &b);

}

#endif  // TINYTORCH_OP_TENSOR_EQ_H_