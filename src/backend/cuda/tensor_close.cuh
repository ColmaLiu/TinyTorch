#ifndef TINYTORCH_BACKEND_CUDA_TENSOR_CLOSE_CUH_
#define TINYTORCH_BACKEND_CUDA_TENSOR_CLOSE_CUH_

namespace TinyTorch::Backend::CUDA {

bool tensor_close(float *a, float *b, int n);

}

#endif  // TINYTORCH_BACKEND_CUDA_TENSOR_CLOSE_CUH_