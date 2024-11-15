#ifndef TINYTORCH_BACKEND_CUDA_TENSOR_SCALAR_OP_CUH_
#define TINYTORCH_BACKEND_CUDA_TENSOR_SCALAR_OP_CUH_

namespace TinyTorch::Backend::CUDA {

void tensor_adds(float *input, float *output, float scalar, int n);

void tensor_subs(float *input, float *output, float scalar, int n);

void tensor_muls(float *input, float *output, float scalar, int n);

void tensor_divs(float *input, float *output, float scalar, int n);

void tensor_pows(float *input, float *output, float scalar, int n);

}

#endif  // TINYTORCH_BACKEND_CUDA_TENSOR_SCALAR_OP_CUH_