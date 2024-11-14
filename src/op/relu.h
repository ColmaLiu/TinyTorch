#ifndef TINYTORCH_OP_RELU_H_
#define TINYTORCH_OP_RELU_H_

#include "tensor/tensor.cuh"

namespace TinyTorch {

// params: input
// return: output
Tensor relu_forward(const Tensor &input);

// params: input, grad_output
// return: grad_input
Tensor relu_backward(const Tensor &input, const Tensor &grad_output);

}

#endif  // TINYTORCH_OP_RELU_H_