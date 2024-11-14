#include "backend/cuda/conv2d.cuh"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>

#include "backend/cuda/utils.cuh"

namespace TinyTorch::Backend::CUDA {

void conv2d_forward(float *input, float *output, float *kernel, float *bias, int batchsize, int channels_in, int channels_out,
                    int height, int width, int ksize, int pad, int stride) {
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    thrust::device_vector<float> col(channels_in * ksize * ksize * height_col * width_col);
    for (int i = 0; i < batchsize; i++) {
        im2col(input + channels_in * height * width * i, channels_in, height, width, ksize, pad, stride,
                   thrust::raw_pointer_cast(col.data()));
        cudaDeviceSynchronize();
        gemm(CUBLAS_OP_N, CUBLAS_OP_N, channels_out, height_col * width_col, channels_in * ksize * ksize,
                 kernel, thrust::raw_pointer_cast(col.data()), output + channels_out * height_col * width_col * i);
        mav2(bias, output + channels_out * height_col * width_col * i, channels_out, height_col * width_col);
    }
}

void conv2d_backward(float *input, float *output, float *kernel, float *bias, int batchsize, int channels_in, int channels_out,
                     int height, int width, int ksize, int pad, int stride, float *grad_output, float *grad_input,
                     float *grad_kernel, float *grad_bias) {
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    thrust::device_vector<float> grad_col(channels_in * ksize * ksize * height_col * width_col);
    thrust::device_vector<float> col(channels_in * ksize * ksize * height_col * width_col);
    thrust::device_ptr<float> grad_in(grad_input);
    thrust::device_ptr<float> grad_out(grad_output);
    thrust::device_vector<float> grad_bias_(channels_out);
    // kernel   cout, cin, k, k
    // output   n, cout, hc, wc
    // col      cin, k, k, hc, wc
    // input    n, cin, h, w
    for (int i = 0; i < batchsize; i++) {
        gemm(CUBLAS_OP_T, CUBLAS_OP_N, channels_in * ksize * ksize, height_col * width_col, channels_out, kernel,
                 grad_output + channels_out * height_col * width_col * i, thrust::raw_pointer_cast(grad_col.data()));
        col2im(grad_input + channels_in * height * width * i, channels_in, height, width, ksize, pad, stride,
                   thrust::raw_pointer_cast(grad_col.data()));
        cudaDeviceSynchronize();
        // calc grad_kernel
        im2col(input + channels_in * height * width * i, channels_in, height, width, ksize, pad, stride,
                   thrust::raw_pointer_cast(col.data()));
        cudaDeviceSynchronize();
        gemm(CUBLAS_OP_N, CUBLAS_OP_T, channels_out, channels_in * ksize * ksize, height_col * width_col,
                 grad_output + channels_out * height_col * width_col * i, thrust::raw_pointer_cast(col.data()), grad_kernel, 1.0f, 1.0f);
        // calc grad_bias
        for (int j = 0; j < channels_out; j++) {
            grad_bias_[j] += thrust::reduce(grad_out + channels_out * height_col * width_col * i + height_col * width_col * j,
                           grad_out + channels_out * height_col * width_col * i + height_col * width_col * (j + 1),
                           0.0f, thrust::plus<float>());
        }
    }
    thrust::copy(grad_bias_.begin(), grad_bias_.end(), grad_bias);
}

}