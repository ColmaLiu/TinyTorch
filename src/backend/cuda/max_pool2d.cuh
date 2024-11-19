#ifndef TINYTORCH_BACKEND_CUDA_MAX_POOL2D_CUH_
#define TINYTORCH_BACKEND_CUDA_MAX_POOL2D_CUH_

namespace TinyTorch::Backend::CUDA {

void max_pool2d_forward(float *input, float *output, float *mask, int batchsize, int channels, int height, int width,
                           int ksize, int pad, int stride, int height_out, int width_out);

void max_pool2d_backward(float *input, float *output, float *mask, int batchsize, int channels, int height, int width,
                            int ksize, int pad, int stride, int height_out, int width_out,
                            float *grad_output, float *grad_input);

}

#endif  // TINYTORCH_BACKEND_CUDA_MAX_POOL2D_CUH_