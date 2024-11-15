#ifndef TINYTORCH_UTILS_UTILS_H_
#define TINYTORCH_UTILS_UTILS_H_

#include <vector>

namespace TinyTorch {

inline int get_product_over_vector(const std::vector<int> vec) {
    int res = 1;
    for (auto &i : vec) {
        res *= i;
    }
    return res;
}

}

#endif  // TINYTORCH_UTILS_UTILS_H_