#ifndef TINYTORCH_OP_CROSS_ENTROPY_H_
#define TINYTORCH_OP_CROSS_ENTROPY_H_

#include <tuple>

#include "tensor/tensor.cuh"

namespace TinyTorch {

// params: input, target
// return: prob, loss
std::tuple<Tensor, Tensor> cross_entropy_forward(const Tensor &input, const Tensor &target);

// params: prob, target
// return: grad_input
Tensor cross_entropy_backward(const Tensor &prob, const Tensor &target);

}

#endif  // TINYTORCH_OP_CROSS_ENTROPY_H_