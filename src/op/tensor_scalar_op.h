#ifndef TINYTORCH_OP_TENSOR_SCALAR_OP_H_
#define TINYTORCH_OP_TENSOR_SCALAR_OP_H_

#include "tensor/tensor.cuh"

namespace TinyTorch {

Tensor tensor_adds(const Tensor &input, const float scalar);

Tensor tensor_subs(const Tensor &input, const float scalar);

Tensor tensor_muls(const Tensor &input, const float scalar);

Tensor tensor_divs(const Tensor &input, const float scalar);

Tensor tensor_pows(const Tensor &input, const float scalar);

}

#endif  // TINYTORCH_OP_TENSOR_SCALAR_OP_H_