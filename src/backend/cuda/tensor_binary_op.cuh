#ifndef TINYTORCH_BACKEND_CUDA_TENSOR_BINARY_OP_CUH_
#define TINYTORCH_BACKEND_CUDA_TENSOR_BINARY_OP_CUH_

namespace TinyTorch::Backend::CUDA {

void tensor_add(float *a, float *b, float *res, int n);

void tensor_sub(float *a, float *b, float *res, int n);

void tensor_mul(float *a, float *b, float *res, int n);

void tensor_div(float *a, float *b, float *res, int n);

}

#endif  // TINYTORCH_BACKEND_CUDA_TENSOR_BINARY_OP_CUH_