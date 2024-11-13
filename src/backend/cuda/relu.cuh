#ifndef TINYTORCH_BACKEND_CUDA_RELU_CUH_
#define TINYTORCH_BACKEND_CUDA_RELU_CUH_

namespace TinyTorch::Backend::CUDA {

void relu_forward(float* in, float* out, int n);

void relu_backward(float *grad_out, float *in, float *grad_in, int n);

}

#endif  // TINYTORCH_BACKEND_CUDA_RELU_CUH_