#ifndef TINYTORCH_OP_LINEAR_H_
#define TINYTORCH_OP_LINEAR_H_

#include <tuple>

#include "tensor/tensor.cuh"

namespace TinyTorch {

// params: input, weight, bias
// return: output
Tensor linear_forward(const Tensor &input, const Tensor &weight, const Tensor &bias);

// params: input, weight, grad_output
// return: grad_input, grad_weight, grad_bias
std::tuple<Tensor, Tensor, Tensor> linear_backward(
        const Tensor &input,
        const Tensor &weight,
        const Tensor &grad_output);

}

#endif  // TINYTORCH_OP_LINEAR_H_