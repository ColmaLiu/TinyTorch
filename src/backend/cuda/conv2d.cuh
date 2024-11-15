#ifndef TINYTORCH_BACKEND_CUDA_CONV2D_CUH_
#define TINYTORCH_BACKEND_CUDA_CONV2D_CUH_

namespace TinyTorch::Backend::CUDA {

void conv2d_forward(float *input, float *output, float *kernel, float *bias, int batchsize, int channels_in, int channels_out,
                    int height, int width, int ksize, int pad, int stride, int height_col, int width_col);

void conv2d_backward(float *input, float *output, float *kernel, float *bias, int batchsize, int channels_in, int channels_out,
                     int height, int width, int ksize, int pad, int stride, int height_col, int width_col,
                     float *grad_output, float *grad_input,float *grad_kernel, float *grad_bias);

}

#endif  // TINYTORCH_BACKEND_CUDA_CONV2D_CUH_