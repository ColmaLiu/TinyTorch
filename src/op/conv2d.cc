#include "op/conv2d.h"

#include <tuple>

#include "utils/utils.h"
#include "tensor/tensor.cuh"
#include "backend/cuda/conv2d.cuh"

namespace TinyTorch {

Tensor conv2d_forward(const Tensor &input, const Tensor &weight, const Tensor &bias, int stride, int padding) {
    ASSERT(input.device.is_cuda() &&
           weight.device.is_cuda() &&
           bias.device.is_cuda());
    ASSERT(input.dim() == 4 &&
           weight.dim() == 4 && 
           bias.dim() == 1);
    int batchsize = input.shape[0];
    int channels_in = weight.shape[1];
    int channels_out = weight.shape[0];
    int height = input.shape[2];
    int width = input.shape[3];
    int kernel_size = weight.shape[2];
    ASSERT(weight.shape[3] == kernel_size &&
           input.shape[1] == channels_in &&
           bias.shape[0] == channels_out);
    int height_out = (height + 2 * padding - kernel_size) / stride + 1;
    int width_out = (width + 2 * padding - kernel_size) / stride + 1;
    Tensor output({batchsize, channels_out, height_out, width_out}, input.device);
    if (input.device.is_cuda()) {
        Backend::CUDA::conv2d_forward(input.data, output.data, weight.data, bias.data,
                                      batchsize, channels_in, channels_out, height, width,
                                      kernel_size, padding, stride, height_out, width_out);
    } else {}
    return output;
}

std::tuple<Tensor, Tensor, Tensor> conv2d_backward(const Tensor &input, const Tensor &weight, const Tensor &grad_output,
                       int stride, int padding) {
    ASSERT(input.device.is_cuda() &&
           weight.device.is_cuda() &&
           grad_output.device.is_cuda());
    ASSERT(input.dim() == 4 &&
           weight.dim() == 4 &&
           grad_output.dim() == 4);
    int batchsize = input.shape[0];
    int channels_in = weight.shape[1];
    int channels_out = weight.shape[0];
    int height = input.shape[2];
    int width = input.shape[3];
    int kernel_size = weight.shape[2];
    int height_out = (height + 2 * padding - kernel_size) / stride + 1;
    int width_out = (width + 2 * padding - kernel_size) / stride + 1;
    ASSERT(weight.shape[3] == kernel_size &&
           input.shape[1] == channels_in &&
           grad_output.shape[0] == batchsize &&
           grad_output.shape[1] == channels_out &&
           grad_output.shape[2] == height_out &&
           grad_output.shape[3] == width_out);
    Tensor grad_input(input.shape, input.device);
    Tensor grad_weight(weight.shape, input.device);
    Tensor grad_bias({channels_out}, input.device);
    if (input.device.is_cuda()) {
        Backend::CUDA::conv2d_backward(input.data, nullptr, weight.data, nullptr,
                                       batchsize, channels_in, channels_out, height, width,
                                       kernel_size, padding, stride, height_out, width_out,
                                       grad_output.data, grad_input.data, grad_weight.data, grad_bias.data);
    } else {}
    return std::make_tuple(std::move(grad_input), std::move(grad_weight), std::move(grad_bias));
}

}