#include "tensor/tensor.cuh"

#include <cstddef>
#include <vector>

#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>

#include "basic/device.h"
#include "basic/mem.cuh"
#include "op/tensor_binary_op.h"
#include "op/tensor_eq.h"
#include "op/tensor_scalar_op.h"
#include "op/tensor_unary_op.h"
#include "utils/utils.h"

namespace TinyTorch {

inline float *Tensor::allocate_cpu() {
    float *data_ = new float[numel()];
    return data_;
}
inline float *Tensor::allocate_cuda() {
    float *data_ = nullptr;
    cudaMalloc(&data_, numel() * sizeof(float));
    return data_;
}

inline void Tensor::free_data() {
    if (this->device.is_cpu()) {
        delete[] this->data;
    } else if (this->device.is_cuda()) {
        cudaFree(this->data);
    } else {}
}

Tensor::Tensor(const std::vector<int> &shape, Device device)
        : shape(shape), device(device) {
    if (device.is_cpu()) {
        data = allocate_cpu();
    } else if (device.is_cuda()) {
        data = allocate_cuda();
    } else {}
}
Tensor::Tensor(Tensor &&other) noexcept
        : shape(other.shape), device(other.device), data(other.data) {
    other.data = nullptr;
}
Tensor &Tensor::operator=(Tensor &&other) noexcept {
    if (this != &other) {
        if (data) {
            free_data();
        } else {}
        data = other.data;
        other.data = nullptr;
        shape = other.shape;
        device = other.device;
    } else {}
    return *this;
}
Tensor::Tensor(const Tensor &other)
        : shape(other.shape), device(other.device) {
    if (device.is_cpu()) {
        data = allocate_cpu();
    } else if (device.is_cuda()) {
        data = allocate_cuda();
    } else {}
    memcpy(data, device, other.data, other.device, numel() * sizeof(float));
}
Tensor &Tensor::operator=(const Tensor &other) {
    if (this != &other) {
        if (data) {
            free_data();
        } else {}
        shape = other.shape;
        device = other.device;
        if (device.is_cpu()) {
            data = allocate_cpu();
        } else if (device.is_cuda()) {
            data = allocate_cuda();
        } else {}
        memcpy(data, device, other.data, other.device, numel() * sizeof(float));
    } else {}
    return *this;
}

Tensor Tensor::to(Device target) const {
    Tensor ret(shape, target);
    memcpy(ret.data, ret.device, data, device, numel() * sizeof(float));
    return ret;
}
Tensor Tensor::cpu() const {
    return to(Device::cpu());
}
Tensor Tensor::cuda() const {
    return to(Device::cuda());
}

int Tensor::numel() const {
    return get_product_over_vector(shape);
}
float *Tensor::data_ptr() const {
    return data;
}
int Tensor::dim() const {
    return shape.size();
}

Tensor Tensor::from_vector(const std::vector<float> &data, const std::vector<int> &shape, Device device) {
    ASSERT(data.size() == get_product_over_vector(shape));
    Tensor ret(shape, Device::cpu());
    for (int i = 0; i < data.size(); i++) {
        ret.data[i] = data[i];
    }
    return ret.to(device);
}

Tensor Tensor::zeros(const std::vector<int> &shape, Device device) {
    return fill(0, shape, device);
}
Tensor Tensor::zeros_like(const Tensor &other) {
    return zeros(other.shape, other.device);
}
Tensor Tensor::fill(const float &scalar, const std::vector<int> &shape, Device device) {
    Tensor ret(shape, device);
    if (device.is_cuda()) {
        thrust::device_ptr<float> data(ret.data);
        thrust::fill(data, data + ret.numel(), scalar);
    } else if (device.is_cpu()) {
        thrust::fill(ret.data, ret.data + ret.numel(), scalar);
    } else {}
    return ret;
}
Tensor Tensor::fill_like(const float &scalar, const Tensor &other) {
    return fill(scalar, other.shape, other.device);
}

Tensor::~Tensor() {
    free_data();
}

bool Tensor::operator==(const Tensor &other) const {
    return tensor_eq(*this, other);
}
bool Tensor::operator!=(const Tensor &other) const {
    return !tensor_eq(*this, other);
}

Tensor Tensor::operator+(const Tensor &other) const {
    return tensor_add(*this, other);
}
Tensor Tensor::operator-(const Tensor &other) const {
    return tensor_sub(*this, other);
}
Tensor Tensor::operator*(const Tensor &other) const {
    return tensor_mul(*this, other);
}
Tensor Tensor::operator/(const Tensor &other) const {
    return tensor_div(*this, other);
}

Tensor Tensor::operator+(const float &scalar) const {
    return tensor_adds(*this, scalar);
}
Tensor Tensor::operator-(const float &scalar) const {
    return tensor_subs(*this, scalar);
}
Tensor Tensor::operator*(const float &scalar) const {
    return tensor_muls(*this, scalar);
}
Tensor Tensor::operator/(const float &scalar) const {
    return tensor_divs(*this, scalar);
}

Tensor Tensor::operator-() const {
    return tensor_neg(*this);
}

}