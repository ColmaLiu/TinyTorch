#ifndef TINYTORCH_BACKEND_CUDA_LINEAR_CUH_
#define TINYTORCH_BACKEND_CUDA_LINEAR_CUH_

namespace TinyTorch::Backend::CUDA {

void linear_forward(float *input, float *output, float *weight, float *bias, int batchsize,
                    int in_features, int out_features);

void linear_backward(float *input, float *output, float *weight, float *bias, int batchsize,
                     int in_features, int out_features, float *grad_output, float *grad_input,
                     float *grad_weights, float *grad_bias);

}

#endif  // TINYTORCH_BACKEND_CUDA_LINEAR_CUH_