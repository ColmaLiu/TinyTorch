#include "backend/cuda/tensor_binary_op.cuh"

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>

#include "backend/cuda/utils.cuh"

namespace TinyTorch::Backend::CUDA {

void tensor_add(float *a, float *b, float *res, int n) {
    thrust::device_ptr<float> a_(a);
    thrust::device_ptr<float> b_(b);
    thrust::device_ptr<float> res_(res);
    thrust::transform(a_, a_ + n, b_, res_, thrust::plus<float>());
}

void tensor_sub(float *a, float *b, float *res, int n) {
    thrust::device_ptr<float> a_(a);
    thrust::device_ptr<float> b_(b);
    thrust::device_ptr<float> res_(res);
    thrust::transform(a_, a_ + n, b_, res_, thrust::minus<float>());
}

void tensor_mul(float *a, float *b, float *res, int n) {
    thrust::device_ptr<float> a_(a);
    thrust::device_ptr<float> b_(b);
    thrust::device_ptr<float> res_(res);
    thrust::transform(a_, a_ + n, b_, res_, thrust::multiplies<float>());
}

void tensor_div(float *a, float *b, float *res, int n) {
    thrust::device_ptr<float> a_(a);
    thrust::device_ptr<float> b_(b);
    thrust::device_ptr<float> res_(res);
    thrust::transform(a_, a_ + n, b_, res_, thrust::divides<float>());
}

}