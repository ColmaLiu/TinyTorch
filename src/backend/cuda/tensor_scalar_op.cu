#include "backend/cuda/tensor_scalar_op.cuh"

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>

#include "backend/cuda/utils.cuh"

namespace TinyTorch::Backend::CUDA {

void tensor_adds(float *input, float *output, float scalar, int n) {
    thrust::device_ptr<float> in(input);
    thrust::device_ptr<float> out(output);
    thrust::transform(in, in + n, out, add_scalar(scalar));
}

void tensor_subs(float *input, float *output, float scalar, int n) {
    thrust::device_ptr<float> in(input);
    thrust::device_ptr<float> out(output);
    thrust::transform(in, in + n, out, sub_scalar(scalar));
}

void tensor_muls(float *input, float *output, float scalar, int n) {
    thrust::device_ptr<float> in(input);
    thrust::device_ptr<float> out(output);
    thrust::transform(in, in + n, out, mul_scalar(scalar));
}

void tensor_divs(float *input, float *output, float scalar, int n) {
    thrust::device_ptr<float> in(input);
    thrust::device_ptr<float> out(output);
    thrust::transform(in, in + n, out, div_scalar(scalar));
}

void tensor_pows(float *input, float *output, float scalar, int n) {
    thrust::device_ptr<float> in(input);
    thrust::device_ptr<float> out(output);
    thrust::transform(in, in + n, out, pow_scalar(scalar));
}

}