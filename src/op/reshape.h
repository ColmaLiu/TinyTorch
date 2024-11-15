#ifndef TINYTORCH_OP_RESHAPE_H_
#define TINYTORCH_OP_RESHAPE_H_

#include "tensor/tensor.cuh"

namespace TinyTorch {

Tensor reshape(const Tensor &input, const std::vector<int> &shape);

}

#endif  // TINYTORCH_OP_RESHAPE_H_