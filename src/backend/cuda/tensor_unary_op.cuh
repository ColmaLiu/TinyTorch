#ifndef TINYTORCH_BACKEND_CUDA_TENSOR_UNARY_OP_CUH_
#define TINYTORCH_BACKEND_CUDA_TENSOR_UNARY_OP_CUH_

namespace TinyTorch::Backend::CUDA {

void tensor_neg(float *input, float *output, int n);

void tensor_inv(float *input, float *output, int n);

void tensor_exp(float *input, float *output, int n);

void tensor_log(float *input, float *output, int n);

}

#endif  // TINYTORCH_BACKEND_CUDA_TENSOR_UNARY_OP_CUH_