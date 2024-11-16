#include "pybind/basic.h"

#include <pybind11/pybind11.h>

#include "basic/device.h"

using namespace pybind11::literals;
namespace py = pybind11;

void init_basic(py::module &m) {
    py::class_<TinyTorch::Device>(m, "Device")
        .def("__str__", &TinyTorch::Device::to_string)
        .def("is_cpu", &TinyTorch::Device::is_cpu)
        .def("is_cuda", &TinyTorch::Device::is_cuda)
        .def("__eq__", &TinyTorch::Device::operator==)
        .def("__ne__", &TinyTorch::Device::operator!=)
        .def_static("cpu", &TinyTorch::Device::cpu)
        .def_static("cuda", &TinyTorch::Device::cuda)
        .def_static("get_default_device", &TinyTorch::Device::get_default_device)
        .def_static("set_default_device", &TinyTorch::Device::set_default_device, "device"_a);
}