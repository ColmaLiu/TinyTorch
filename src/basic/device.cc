#include "basic/device.h"

#include <string>

#include <cuda_runtime.h>

namespace TinyTorch {

Device::Device(device_type_t type)
        : type(type) {}

std::string Device::to_string() const {
    if (type == device_type_t::CPU) {
        return "cpu";
    } else if (type == device_type_t::CUDA) {
        return "cuda:0";
    } else {
        exit(1);
    }
}

std::string Device::repr() const {

}

Device Device::default_device = Device::cpu();

Device Device::get_default_device() {
    return default_device;
}

}