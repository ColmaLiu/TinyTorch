#include "backend/cuda/max_pooling2d.cuh"

#include <cuda_runtime.h>

#include "backend/cuda/utils.cuh"

namespace TinyTorch::Backend::CUDA {

__global__ void max_pooling2d_forward_kernel(const int n, const float *input, float *output, float *mask,
                                             const int height_in, const int width_in,
                                             const int height_out, const int width_out,
                                             const int ksize, const int pad, const int stride) {
    CUDA_KERNEL_LOOP(index, n) {
        int w_out = index % width_out;
        int h_index = index / width_out;
        int h_out = h_index % height_out;
        int ch = h_index / height_out;
        int h_in = h_out * stride - pad;
        int w_in = w_out * stride - pad;
        int offset = (ch * height_in + h_in) * width_in + w_in;
        const float *data_ptr = input;
        data_ptr += offset;
        int maxi = 0, maxj = 0;
        for (int i = 0; i < ksize; i++) {
            for (int j = 0; j < ksize; ++j) {
                int h = h_in + i;
                int w = w_in + j;
                if (h >= 0 && w >= 0 && h < height_in && w < width_in
                    && data_ptr[i * width_in + j] > data_ptr[maxi * width_in + maxj]) {
                    maxi = i;
                    maxj = j;
                }
            }
        }
        output[index] = data_ptr[maxi * width_in + maxj];
        mask[offset + maxi * width_in + maxj] = 1.0f;
    }
}

__global__ void max_pooling2d_backward_kernel(const int n, const float *input, const float *output, const float *mask,
                                              const int height_in, const int width_in,
                                              const int height_out, const int width_out,
                                              const int ksize, const int pad, const int stride,
                                              float *grad_output, float *grad_input) {
    CUDA_KERNEL_LOOP(index, n) {
        int w_out = index % width_out;
        int h_index = index / width_out;
        int h_out = h_index % height_out;
        int ch = h_index / height_out;
        int h_in = h_out * stride - pad;
        int w_in = w_out * stride - pad;
        int offset = (ch * height_in + h_in) * width_in + w_in;
        const float *mask_ptr = mask;
        mask_ptr += offset;
        for (int i = 0; i < ksize; i++) {
            for (int j = 0; j < ksize; ++j) {
                int h = h_in + i;
                int w = w_in + j;
                if (h >= 0 && w >= 0 && h < height_in && w < width_in
                    && mask_ptr[i * width_in + j] == 1.0f) {
                    grad_input[offset + i * width_in + j] = grad_output[index];
                    break;
                }
            }
        }
    }
}

void max_pooling2d_forward(float *input, float *output, float *mask, int batchsize, int channels, int height, int width,
                           int ksize, int pad, int stride) {
    int height_out = (height + 2 * pad - ksize) / stride + 1;
    int width_out = (width + 2 * pad - ksize) / stride + 1;
    int n = batchsize * channels * height_out * width_out;
    max_pooling2d_forward_kernel<<<CudaGetBlocks(n), kCudaThreadsNum>>>(n, input, output, mask, height, width,
                                                                        height_out, width_out, ksize, pad, stride);
    cudaDeviceSynchronize();
}

void max_pooling2d_backward(float *input, float *output, float *mask, int batchsize, int channels, int height, int width,
                            int ksize, int pad, int stride, float *grad_output, float *grad_input) {
    int height_out = (height + 2 * pad - ksize) / stride + 1;
    int width_out = (width + 2 * pad - ksize) / stride + 1;
    int n = batchsize * channels * height_out * width_out;
    max_pooling2d_backward_kernel<<<CudaGetBlocks(n), kCudaThreadsNum>>>(n, input, output, mask, height, width,
                                                                         height_out, width_out, ksize, pad, stride,
                                                                         grad_output, grad_input);
    cudaDeviceSynchronize();
}

}