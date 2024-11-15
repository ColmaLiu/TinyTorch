#include "op/linear.h"

#include <cassert>
#include <tuple>

#include "tensor/tensor.cuh"
#include "backend/cuda/linear.cuh"

namespace TinyTorch {

Tensor linear_forward(const Tensor &input, const Tensor &weight, const Tensor &bias) {
    assert(input.device.is_cuda() && weight.device.is_cuda() && bias.device.is_cuda());
    assert(input.dim() == 2 &&
           weight.dim() == 2 &&
           bias.dim() == 1);
    assert(input.shape[1] == weight.shape[0] &&
           weight.shape[1] == bias.shape[0]);
    int batchsize = input.shape[0];
    int in_features = weight.shape[0];
    int out_features = weight.shape[1];
    Tensor output({batchsize, out_features}, input.device);
    if (input.device.is_cuda()) {
        Backend::CUDA::linear_forward(input.data, output.data, weight.data, bias.data,
                                      batchsize, in_features, out_features);
    } else {}
    return output;
}

std::tuple<Tensor, Tensor, Tensor> linear_backward(
        const Tensor &input,
        const Tensor &weight,
        const Tensor &grad_output) {
    assert(input.device.is_cuda() &&
           weight.device.is_cuda() &&
           grad_output.device.is_cuda());
    assert(input.dim() == 2 &&
           weight.dim() == 2 &&
           grad_output.dim() == 2);
    assert(input.shape[0] == grad_output.shape[0] &&
           input.shape[1] == weight.shape[0] &&
           grad_output.shape[1] == weight.shape[1]);
    int batchsize = input.shape[0];
    int in_features = weight.shape[0];
    int out_features = weight.shape[1];
    Tensor grad_input(input.shape, input.device);
    Tensor grad_weight(weight.shape, input.device);
    Tensor grad_bias({out_features}, input.device);
    if (input.device.is_cuda()) {
        Backend::CUDA::linear_backward(input.data, nullptr, weight.data, nullptr,
                                       batchsize, in_features, out_features,
                                       grad_output.data, grad_input.data, grad_weight.data, grad_bias.data);
    } else {}
    return std::make_tuple(std::move(grad_input), std::move(grad_weight), std::move(grad_bias));
}

}