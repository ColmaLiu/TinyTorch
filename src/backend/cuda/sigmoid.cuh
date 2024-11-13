#ifndef TINYTORCH_BACKEND_CUDA_SIGMOID_CUH_
#define TINYTORCH_BACKEND_CUDA_SIGMOID_CUH_

namespace TinyTorch::Backend::CUDA {

void sigmoid_forward(float* in, float* out, int n);

void sigmoid_backward(float *grad_out, float *in, float *grad_in, int n);

}

#endif  // TINYTORCH_BACKEND_CUDA_SIGMOID_CUH_