#ifndef TINYTORCH_TERSOR_TENSOR_H_
#define TINYTORCH_TERSOR_TENSOR_H_

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

    ~Tensor();

    std::vector<int> shape;
    Device device;
    float *data;
};

}

#endif  // TINYTORCH_TERSOR_TENSOR_H_
