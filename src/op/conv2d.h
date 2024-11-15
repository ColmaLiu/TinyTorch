#ifndef TINYTORCH_OP_CONV2D_H_
#define TINYTORCH_OP_CONV2D_H_

#include <tuple>

#include "tensor/tensor.cuh"

namespace TinyTorch {

// params: input, weight, bias, stride, padding
// return: output
Tensor conv2d_forward(const Tensor &input, const Tensor &weight, const Tensor &bias, int stride, int padding);

// params: input, weight, grad_output, stride, padding
// return: grad_input, grad_weight, grad_bias
std::tuple<Tensor, Tensor, Tensor> conv2d_backward(const Tensor &input, const Tensor &weight, const Tensor &grad_output,
                       int stride, int padding);

}

#endif  // TINYTORCH_OP_CONV2D_H_