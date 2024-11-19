#ifndef TINYTORCH_OP_MAX_POOL2D_H_
#define TINYTORCH_OP_MAX_POOL2D_H_

#include <tuple>

#include "tensor/tensor.cuh"

namespace TinyTorch {

// params: input, kernel_size, stride, padding
// return: output, mask
std::tuple<Tensor, Tensor> max_pool2d_forward(const Tensor &input, int kernel_size, int stride, int padding);

// params: mask, grad_output, kernel_size, stride, padding
// return: grad_input
Tensor max_pool2d_backward(const Tensor &mask, const Tensor &grad_output, int kernel_size, int stride, int padding);

}

#endif  // TINYTORCH_OP_MAX_POOL2D_H_