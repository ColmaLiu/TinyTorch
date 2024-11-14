#include "backend/cuda/sigmoid.cuh"

#include <cuda_runtime.h>

#include "backend/cuda/utils.cuh"

namespace TinyTorch::Backend::CUDA {

__global__ void sigmoid_forward_kernel(const float *in, float *out, int n) {
    CUDA_KERNEL_LOOP(i, n) {
        out[i] = 1.0f / (1.0f + expf(-in[i]));
    }
}

__global__ void sigmoid_backward_kernel(const float *grad_out, const float *out, float *grad_in, int n) {
    CUDA_KERNEL_LOOP(i, n){
        grad_in[i] = grad_out[i] * out[i] * (1 - out[i]);
    }
}

void sigmoid_forward(float* in, float* out, int n) {
    sigmoid_forward_kernel<<<CudaGetBlocks(n), kCudaThreadsNum>>>(in, out, n);
}

void sigmoid_backward(float *grad_out, float *out, float *grad_in, int n) {
    sigmoid_backward_kernel<<<CudaGetBlocks(n), kCudaThreadsNum>>>(grad_out, out, grad_in, n);
}

}