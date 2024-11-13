#ifndef TINYTORCH_BACKEND_CUDA_UTILS_H_
#define TINYTORCH_BACKEND_CUDA_UTILS_H_

const int kCudaThreadsNum = 512;

inline int CudaGetBlocks(const int N) {
    return (N + kCudaThreadsNum - 1) / kCudaThreadsNum;
}

#define CUDA_KERNEL_LOOP(i, n) \
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

#endif  // TINYTORCH_BACKEND_CUDA_UTILS_H_