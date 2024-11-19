#include "op/max_pool2d.h"

#include <tuple>

#include "utils/utils.h"
#include "tensor/tensor.cuh"
#include "backend/cuda/max_pool2d.cuh"

namespace TinyTorch {

std::tuple<Tensor, Tensor> max_pool2d_forward(const Tensor &input, int kernel_size, int stride, int padding) {
    ASSERT(input.device.is_cuda());
    ASSERT(input.dim() == 4);
    ASSERT(kernel_size == stride);
    int batchsize = input.shape[0];
    int channels = input.shape[1];
    int height = input.shape[2];
    int width = input.shape[3];
    int height_out = (height + 2 * padding - kernel_size) / stride + 1;
    int width_out = (width + 2 * padding - kernel_size) / stride + 1;
    Tensor output({batchsize, channels, height_out, width_out}, input.device);
    Tensor mask(input.shape, input.device);
    if (input.device.is_cuda()) {
        Backend::CUDA::max_pool2d_forward(input.data, output.data, mask.data,
                                             batchsize, channels, height, width,
                                             kernel_size, padding, stride, height_out, width_out);
    } else {}
    return std::make_tuple(std::move(output), std::move(mask));
}

Tensor max_pool2d_backward(const Tensor &mask, const Tensor &grad_output, int kernel_size, int stride, int padding) {
    ASSERT(mask.device.is_cuda() && grad_output.device.is_cuda());
    ASSERT(mask.dim() == 4 && grad_output.dim() == 4);
    ASSERT(mask.shape[0] == grad_output.shape[0] &&
           mask.shape[1] == grad_output.shape[1]);
    ASSERT(kernel_size == stride);
    int batchsize = mask.shape[0];
    int channels = mask.shape[1];
    int height = mask.shape[2];
    int width = mask.shape[3];
    int height_out = (height + 2 * padding - kernel_size) / stride + 1;
    int width_out = (width + 2 * padding - kernel_size) / stride + 1;
    ASSERT(grad_output.shape[2] == height_out &&
           grad_output.shape[3] == width_out);
    Tensor grad_input(mask.shape, mask.device);
    if (mask.device.is_cuda()) {
        Backend::CUDA::max_pool2d_backward(nullptr, nullptr, mask.data,
                                              batchsize, channels, height, width,
                                              kernel_size, padding, stride, height_out, width_out,
                                              grad_output.data, grad_input.data);
    } else {}
    return grad_input;
}

}