#ifndef TINYTORCH_BASIC_MEM_H_
#define TINYTORCH_BASIC_MEM_H_

#include <cstddef>

#include <cuda_runtime.h>

#include "basic/device.h"

namespace TinyTorch {

inline cudaMemcpyKind get_cudaMemcpyKind(Device dst, Device src);

void memcpy(void *dst_ptr, Device dst, void *src_ptr, Device src, size_t length);

}

#endif  // TINYTORCH_BASIC_MEM_H_