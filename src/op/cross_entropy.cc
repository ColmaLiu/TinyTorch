#include "op/cross_entropy.h"

#include <cassert>
#include <tuple>

#include "tensor/tensor.cuh"
#include "backend/cuda/cross_entropy.cuh"

namespace TinyTorch {

std::tuple<Tensor, Tensor> cross_entropy_forward(const Tensor &input, const Tensor &target) {
    assert(input.device.is_cuda() && target.device.is_cuda());
    assert(input.dim() == 2 && target.dim() == 1);
    assert(input.shape[0] == target.shape[0]);
    int batchsize = input.shape[0];
    int labels = input.shape[1];
    Tensor prob(input.shape, input.device);
    Tensor loss({}, input.device);
    if (input.device.is_cuda()) {
        Backend::CUDA::softmax_forward(input.data, prob.data,
                                       batchsize, labels);
        Backend::CUDA::cross_entropy_forward(prob.data, target.data, loss.data,
                                       batchsize, labels);
    } else {}
    return std::make_tuple(std::move(prob), std::move(loss));
}

Tensor cross_entropy_backward(const Tensor &prob, const Tensor &target) {
    assert(prob.device.is_cuda() && target.device.is_cuda());
    assert(prob.dim() == 2 && target.dim() == 1);
    assert(prob.shape[0] == target.shape[0]);
    int batchsize = prob.shape[0];
    int labels = prob.shape[1];
    Tensor grad_input(prob.shape, prob.device);
    if (prob.device.is_cuda()) {
        Backend::CUDA::cross_entropy_with_softmax_backward(nullptr, prob.data, target.data, nullptr,
                                                           batchsize, labels,
                                                           grad_input.data);
    } else {}
    return grad_input;
}

}