#include <pybind11/pybind11.h>

#include "pybind/basic.h"
#include "pybind/op.h"
#include "pybind/tensor.h"

PYBIND11_MODULE(tinytorch, m) {
    m.doc() = "TinyTorch: An AI framework written in C++";

    init_basic(m);
    init_op(m);
    init_tensor(m);
}