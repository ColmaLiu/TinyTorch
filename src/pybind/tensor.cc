#include "pybind/tensor.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "tensor/tensor.cuh"
#include "op/reshape.h"

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

        .def_static("from_vector", &Tensor::from_vector, "data"_a, "shape"_a, "device"_a=Device::default_device)

        .def_static("zeros", &Tensor::zeros, "shape"_a, "device"_a=Device::default_device)
        .def_static("zeros_like", &Tensor::zeros_like, "other"_a)
        .def_static("fill", &Tensor::fill, "scalar"_a, "shape"_a, "device"_a=Device::default_device)
        .def_static("fill_like", &Tensor::fill_like, "scalar"_a, "other"_a)

        .def_static("rand", &Tensor::rand, "shape"_a, "device"_a=Device::default_device)
        .def_static("rand_like", &Tensor::rand_like, "other"_a)
        .def_static("randn", &Tensor::rand, "shape"_a, "device"_a=Device::default_device)
        .def_static("randn_like", &Tensor::rand_like, "other"_a)

        .def("__eq__", &Tensor::operator==)
        .def("__ne__", &Tensor::operator!=)

        .def("__add__", static_cast<Tensor (Tensor::*)(const Tensor &) const>(&Tensor::operator+))
        .def("__sub__", static_cast<Tensor (Tensor::*)(const Tensor &) const>(&Tensor::operator-))
        .def("__mul__", static_cast<Tensor (Tensor::*)(const Tensor &) const>(&Tensor::operator*))
        .def("__truediv__", static_cast<Tensor (Tensor::*)(const Tensor &) const>(&Tensor::operator/))

        .def("__add__", static_cast<Tensor (Tensor::*)(const float &) const>(&Tensor::operator+))
        .def("__sub__", static_cast<Tensor (Tensor::*)(const float &) const>(&Tensor::operator-))
        .def("__mul__", static_cast<Tensor (Tensor::*)(const float &) const>(&Tensor::operator*))
        .def("__truediv__", static_cast<Tensor (Tensor::*)(const float &) const>(&Tensor::operator/))

        .def("__neg__", static_cast<Tensor (Tensor::*)() const>(&Tensor::operator-))

        .def(pybind11::init([](pybind11::array_t<float> &array, std::optional<Device> device_) {
            Device device = device_.value_or(Device::get_default_device());
            std::vector<int> shape;
            for (int i = 0; i < array.ndim(); i++) {
                shape.push_back(array.shape(i));
            }
            int numel = array.size();
            std::vector<float> data;
            pybind11::array_t<float> arrat_1d = array.reshape({numel});
            auto accessor = arrat_1d.unchecked<1>();
            for (int i = 0; i < numel; i++) {
                data.push_back(accessor(i));
            }
            return new Tensor(Tensor::from_vector(data, shape, device));
        }), "array"_a, "device"_a = std::optional<Device>())
        .def(pybind11::init([](float &scalar, std::optional<Device> device_) {
            Device device = device_.value_or(Device::get_default_device());
            return new Tensor(Tensor::from_vector({scalar}, {}, device));
        }), "scalar"_a, "device"_a = std::optional<Device>())

        .def("numpy", [](Tensor &instance) {
            int numel = instance.numel();
            Tensor instance_1d = reshape(instance, {numel}).cpu();
            pybind11::array_t<float> array(numel);
            auto accessor = array.mutable_unchecked<1>();
            for (int i = 0; i < numel; i++) {
                accessor(i) = instance_1d.data[i];
            }
            array = array.reshape(instance.shape);
            return array;
        })

        .def_readonly("shape", &Tensor::shape)
        .def_readonly("device", &Tensor::device);
}