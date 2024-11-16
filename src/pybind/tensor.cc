#include "pybind/tensor.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tensor/tensor.cuh"

using namespace TinyTorch;
using namespace pybind11::literals;
namespace py = pybind11;

void init_tensor(pybind11::module &m) {
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<const std::vector<int> &, Device>(),
             "shape"_a, "device"_a = Device::default_device)
        .def(py::init<const Tensor &>())

        .def("to", &Tensor::to, "target"_a)
        .def("cpu", &Tensor::cpu)
        .def("cuda", &Tensor::cuda)

        .def("numel", &Tensor::numel)
        .def("dim", &Tensor::dim)

        .def_readonly("shape", &Tensor::shape)
        .def_readonly("device", &Tensor::device);
}