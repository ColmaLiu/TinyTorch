#include "pybind/op.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "op/conv2d.h"
#include "op/cross_entropy.h"
#include "op/linear.h"
#include "op/max_pool2d.h"
#include "op/relu.h"
#include "op/reshape.h"
#include "op/sigmoid.h"
#include "op/tensor_binary_op.h"
#include "op/tensor_close.h"
#include "op/tensor_eq.h"
#include "op/tensor_scalar_op.h"
#include "op/tensor_unary_op.h"

using namespace TinyTorch;
using namespace pybind11::literals;
namespace py = pybind11;

void init_op(pybind11::module &m) {
    auto op = m.def_submodule("op", "TinyTorch operators");

    op.def("conv2d_forward", &conv2d_forward,
           "params: input, weight, bias, stride, padding\n"
           "return: output",
           "input"_a, "weight"_a, "bias"_a, "stride"_a, "padding"_a);
    op.def("conv2d_backward", &conv2d_backward,
           "params: input, weight, grad_output, stride, padding\n"
           "return: grad_input, grad_weight, grad_bias",
           "input"_a, "weight"_a, "grad_output"_a, "stride"_a, "padding"_a);

    op.def("cross_entropy_forward", &cross_entropy_forward,
           "params: input, target\n"
           "return: prob, loss",
           "input"_a, "target"_a);
    op.def("cross_entropy_backward", &cross_entropy_backward,
           "params: prob, target\n"
           "return: grad_input",
           "prob"_a, "target"_a);

    op.def("linear_forward", &linear_forward,
           "params: input, weight, bias\n"
           "return: output",
           "input"_a, "weight"_a, "bias"_a);
    op.def("linear_backward", &linear_backward,
           "params: input, weight, grad_output\n"
           "return: grad_input, grad_weight, grad_bias",
           "input"_a, "weight"_a, "grad_output"_a);

    op.def("max_pool2d_forward", &max_pool2d_forward,
           "params: input, kernel_size, stride, padding\n"
           "return: output, mask",
           "input"_a, "kernel_size"_a, "stride"_a, "padding"_a);
    op.def("max_pool2d_backward", &max_pool2d_backward,
           "params: mask, grad_output, kernel_size, stride, padding\n"
           "return: grad_input\n"
           "**stride should be equal to kernel_size**",
           "mask"_a, "grad_output"_a, "kernel_size"_a, "stride"_a, "padding"_a);

    op.def("relu_forward", &relu_forward,
           "params: input\n"
           "return: output",
           "input"_a);
    op.def("relu_backward", &relu_backward,
           "params: input, grad_output\n"
           "return: grad_input",
           "input"_a, "grad_output"_a);

    op.def("reshape", &reshape, "input"_a, "shape"_a);

    op.def("sigmoid_forward", &sigmoid_forward,
           "params: input\n"
           "return: output",
           "input"_a);
    op.def("sigmoid_backward", &sigmoid_backward,
           "params: output, grad_output\n"
           "return: grad_input",
           "output"_a, "grad_output"_a);

    op.def("tensor_add", &tensor_add, "a"_a, "b"_a);
    op.def("tensor_sub", &tensor_sub, "a"_a, "b"_a);
    op.def("tensor_mul", &tensor_mul, "a"_a, "b"_a);
    op.def("tensor_div", &tensor_div, "a"_a, "b"_a);

    op.def("tensor_close", &tensor_close, "a"_a, "b"_a);

    op.def("tensor_eq", &tensor_eq, "a"_a, "b"_a);

    op.def("tensor_adds", &tensor_adds, "input"_a, "scalar"_a);
    op.def("tensor_subs", &tensor_subs, "input"_a, "scalar"_a);
    op.def("tensor_muls", &tensor_muls, "input"_a, "scalar"_a);
    op.def("tensor_divs", &tensor_divs, "input"_a, "scalar"_a);
    op.def("tensor_pows", &tensor_pows, "input"_a, "scalar"_a);

    op.def("tensor_neg", &tensor_neg, "input"_a);
    op.def("tensor_inv", &tensor_inv, "input"_a);
    op.def("tensor_exp", &tensor_exp, "input"_a);
    op.def("tensor_log", &tensor_log, "input"_a);
}