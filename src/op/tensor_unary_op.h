#ifndef TINYTORCH_OP_TENSOR_UNARY_OP_H_
#define TINYTORCH_OP_TENSOR_UNARY_OP_H_

#include "tensor/tensor.cuh"

namespace TinyTorch {

Tensor tensor_neg(const Tensor &input);

Tensor tensor_inv(const Tensor &input);

Tensor tensor_exp(const Tensor &input);

Tensor tensor_log(const Tensor &input);

}

#endif  // TINYTORCH_OP_TENSOR_UNARY_OP_H_