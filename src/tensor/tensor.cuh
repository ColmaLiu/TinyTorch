#ifndef TINYTORCH_TERSOR_TENSOR_CUH_
#define TINYTORCH_TERSOR_TENSOR_CUH_

#include <vector>

#include "basic/device.h"

namespace TinyTorch {

class Tensor {
private:
    inline float *allocate_cpu();
    inline float *allocate_cuda();
    inline void free_data();
public:
    Tensor(const std::vector<int> &shape, Device device = Device::default_device);
    Tensor(Tensor &&other) noexcept;
    Tensor &operator=(Tensor &&other) noexcept;
    Tensor(const Tensor &other);
    Tensor &operator=(const Tensor &other);

    Tensor to(Device target) const;
    Tensor cpu() const;
    Tensor cuda() const;

    int numel() const;
    float *data_ptr() const;
    int dim() const;

    static Tensor from_vector(const std::vector<float> &data, const std::vector<int> &shape, Device device);

    ~Tensor();

    bool operator==(const Tensor &other) const;
    bool operator!=(const Tensor &other) const;

    Tensor operator+(const Tensor &other) const;
    Tensor operator-(const Tensor &other) const;
    Tensor operator*(const Tensor &other) const;
    Tensor operator/(const Tensor &other) const;

    Tensor operator+(const float &scalar) const;
    Tensor operator-(const float &scalar) const;
    Tensor operator*(const float &scalar) const;
    Tensor operator/(const float &scalar) const;

    Tensor operator-() const;

    std::vector<int> shape;
    Device device;
    float *data;
};

}

#endif  // TINYTORCH_TERSOR_TENSOR_CUH_