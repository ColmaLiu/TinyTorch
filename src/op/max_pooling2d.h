#ifndef TINYTORCH_OP_MAX_POOLING2D_H_
#define TINYTORCH_OP_MAX_POOLING2D_H_

#include <tuple>

#include "tensor/tensor.cuh"

namespace TinyTorch {

// params: input
// return: output, mask
std::tuple<Tensor, Tensor> max_pooling2d_forward(const Tensor &input, int ksize, int pad, int stride);

// params: mask, grad_output
// return: grad_input
Tensor max_pooling2d_backward(const Tensor &mask, const Tensor &grad_output, int ksize, int pad, int stride);

}

#endif  // TINYTORCH_OP_MAX_POOLING2D_H_