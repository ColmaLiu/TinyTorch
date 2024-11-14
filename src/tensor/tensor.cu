#include "tensor/tensor.cuh"

#include <cstddef>
#include <vector>

#include <cuda_runtime.h>

#include "basic/device.h"
#include "basic/mem.cuh"

namespace TinyTorch {

inline int get_product_over_vector(const std::vector<int> vec) {
    int res = 1;
    for (auto &i : vec) {
        res *= i;
    }
    return res;
}

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

Tensor::Tensor(const std::vector<int> &shape, Device device = Device::default_device)
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

Tensor::~Tensor() {
    free_data();
}

}