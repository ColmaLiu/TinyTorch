#ifndef TINITORCH_TERSOR_TENSOR_H_
#define TINITORCH_TERSOR_TENSOR_H_

#include <inttypes.h>

#include <cstddef>
#include <vector>

class Tensor {
private:
    float *allocate_cpu();
    float *allocate_gpu();
public:
    std::vector<int> shape;
    bool host;
    size_t size;
};

#endif  // TINITORCH_TERSOR_TENSOR_H_
