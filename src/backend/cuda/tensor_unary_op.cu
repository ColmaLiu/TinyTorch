#include "backend/cuda/tensor_unary_op.cuh"

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>

#include "backend/cuda/utils.cuh"

namespace TinyTorch::Backend::CUDA {

void tensor_neg(float *input, float *output, int n) {
    thrust::device_ptr<float> in(input);
    thrust::device_ptr<float> out(output);
    thrust::transform(in, in + n, out, thrust::negate<float>());
}

void tensor_inv(float *input, float *output, int n) {
    thrust::device_ptr<float> in(input);
    thrust::device_ptr<float> out(output);
    thrust::transform(in, in + n, out, inv());
}

void tensor_exp(float *input, float *output, int n) {
    thrust::device_ptr<float> in(input);
    thrust::device_ptr<float> out(output);
    thrust::transform(in, in + n, out, expo());
}

void tensor_log(float *input, float *output, int n) {
    thrust::device_ptr<float> in(input);
    thrust::device_ptr<float> out(output);
    thrust::transform(in, in + n, out, log());
}

}