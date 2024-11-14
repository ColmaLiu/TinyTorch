#ifndef TINYTORCH_BACKEND_CUDA_CROSS_ENTROPY_CUH_
#define TINYTORCH_BACKEND_CUDA_CROSS_ENTROPY_CUH_

namespace TinyTorch::Backend::CUDA {

void softmax_forward(float *input, float *output, int batchsize, int labels);

void cross_entropy_forward(float *input, float *target, float *loss, int batchsize, int labels);

void cross_entropy_with_softmax_backward(float *input, float *prob, float *target, float *loss, int batchsize, int labels,
                                         float *grad_input);

}

#endif  // TINYTORCH_BACKEND_CUDA_CROSS_ENTROPY_CUH_