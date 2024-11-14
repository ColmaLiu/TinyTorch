#include "backend/cuda/cross_entropy.cuh"

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>

#include "backend/cuda/utils.cuh"

namespace TinyTorch::Backend::CUDA {

void softmax_forward(float *input, float *output, int batchsize, int labels) {
    thrust::device_ptr<float> in(input);
    thrust::device_ptr<float> out(output);
    thrust::copy(in, in + batchsize * labels, out);
    for (int i = 0; i < batchsize; i++) {
        float max = thrust::reduce(out + i * labels, out + (i + 1) * labels, -FLT_MAX, thrust::maximum<float>());
        thrust::transform(out + i * labels, out + (i + 1) * labels, out + i * labels, sub_scalar(max));
        thrust::transform(out + i * labels, out + (i + 1) * labels, out + i * labels, expo());
        float sum = thrust::reduce(out + i * labels, out + (i + 1) * labels, 0.0f, thrust::plus<float>());
        thrust::transform(out + i * labels, out + (i + 1) * labels, out + i * labels, div_scalar(sum));
    }
}

void cross_entropy_forward(float *input, float *target, float *loss, int batchsize, int labels) {
    thrust::device_vector<float> ll(batchsize);
    thrust::device_ptr<float> l(loss);
    take_slice(input, target, thrust::raw_pointer_cast(ll.data()), batchsize, labels);
    thrust::transform(ll.begin(), ll.end(), ll.begin(), neglog());
    *l = thrust::reduce(ll.begin(), ll.end(), 0.0f, thrust::plus<float>()) / batchsize;
}

void cross_entropy_with_softmax_backward(float *input, float *prob, float *target, float *loss, int batchsize, int labels,
                                         float *grad_input) {
    thrust::device_ptr<float> p(prob);
    thrust::device_ptr<float> grad_in(grad_input);
    thrust::copy(p, p + batchsize * labels, grad_in);
    slice_sub1(grad_input, target, batchsize, labels);
    thrust::transform(grad_in, grad_in + batchsize * labels, grad_in, div_scalar(batchsize));
}

}