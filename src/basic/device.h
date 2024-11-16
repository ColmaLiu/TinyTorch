#ifndef TINYTORCH_BASIC_DEVICE_H_
#define TINYTORCH_BASIC_DEVICE_H_

#include <string>

namespace TinyTorch {

enum class device_type_t {
    CPU,
    CUDA
};

class Device {
public:
    Device(device_type_t type);

    std::string to_string() const;

    inline bool is_cpu() const {
        return type == device_type_t::CPU;
    }
    inline bool is_cuda() const {
        return type == device_type_t::CUDA;
    }

    inline static Device cpu() {
        return Device(device_type_t::CPU);
    }
    inline static Device cuda() {
        return Device(device_type_t::CUDA);
    }

    inline bool operator==(const Device& other) const {
        return type == other.type;
    }
    inline bool operator!=(const Device& other) const {
        return !(*this == other);
    }

    static Device default_device;
    static Device get_default_device();
    static void set_default_device(const Device &device);

    device_type_t type;
};

}

#endif  // TINYTORCH_BASIC_DEVICE_H_