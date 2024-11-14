#ifndef TINYTORCH_OP_SIGMOID_H_
#define TINYTORCH_OP_SIGMOID_H_

#include "tensor/tensor.cuh"

namespace TinyTorch {

// params: input
// return: output
Tensor sigmoid_forward(const Tensor &input);

// params: output, grad_output
// return: grad_input
Tensor sigmoid_backward(const Tensor &output, const Tensor &grad_output);

}

#endif  // TINYTORCH_OP_SIGMOID_H_