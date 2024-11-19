#include "backend/cuda/tensor_close.cuh"

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>

#include "backend/cuda/utils.cuh"

namespace TinyTorch::Backend::CUDA {

bool tensor_close(float *a, float *b, int n) {
    thrust::device_ptr<float> a_(a);
    thrust::device_ptr<float> b_(b);
    bool is_close = thrust::equal(a_, a_ + n, b_, close());
    return is_close;
}

}