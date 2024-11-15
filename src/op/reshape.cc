#include "op/reshape.h"

#include <cassert>

#include "tensor/tensor.cuh"
#include "utils/utils.h"

namespace TinyTorch {

Tensor reshape(const Tensor &input, const std::vector<int> &shape) {
    assert(input.numel() == get_product_over_vector(shape));
    Tensor output(input);
    output.shape = shape;
    return output;
}

}