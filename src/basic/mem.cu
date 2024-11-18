#include "basic/mem.cuh"

#include <cstddef>

#include <cuda_runtime.h>

#include "basic/device.h"

namespace TinyTorch {

inline cudaMemcpyKind get_cudaMemcpyKind(Device dst, Device src) {
    if (dst.is_cpu() && src.is_cpu()) {
        return cudaMemcpyHostToHost;
    } else if (dst.is_cpu() && src.is_cuda()) {
        return cudaMemcpyDeviceToHost;
    } else if (dst.is_cuda() && src.is_cpu()) {
        return cudaMemcpyHostToDevice;
    } else if (dst.is_cuda() && src.is_cuda()) {
        return cudaMemcpyDeviceToDevice;
    } else {}
}

void memcpy(void *dst_ptr, Device dst, void *src_ptr, Device src, size_t length) {
    cudaMemcpy(dst_ptr, src_ptr, length, get_cudaMemcpyKind(dst, src));
}

}