#ifndef TINYTORCH_BACKEND_CUDA_TENSOR_EQ_CUH_
#define TINYTORCH_BACKEND_CUDA_TENSOR_EQ_CUH_

namespace TinyTorch::Backend::CUDA {

bool tensor_eq(float *a, float *b, int n);

}

#endif  // TINYTORCH_BACKEND_CUDA_TENSOR_EQ_CUH_