#include "backend/cuda/relu.cuh"

#include <cuda_runtime.h>

#include "backend/cuda/utils.cuh"

namespace TinyTorch::Backend::CUDA {

__global__ void relu_forward_kernel(float *in, float *out, int n) {
    CUDA_KERNEL_LOOP(i, n) {
        out[i] = in[i] > 0 ? in[i] : 0;
    }
}

__global__ void relu_backward_kernel(float *grad_out, float *in, float *grad_in, int n) {
    CUDA_KERNEL_LOOP(i, n){
        grad_in[i] = in[i] > 0 ? grad_out[i] : 0;
    }
}

void relu_forward(float* in, float* out, int n) {
    relu_forward_kernel<<<CudaGetBlocks(n), kCudaThreadsNum>>>(in, out, n);
}

void relu_backward(float *grad_out, float *in, float *grad_in, int n) {
    relu_backward_kernel<<<CudaGetBlocks(n), kCudaThreadsNum>>>(grad_out, in, grad_in, n);
}

}