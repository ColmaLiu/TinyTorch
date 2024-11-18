#ifndef TINYTORCH_UTILS_UTILS_H_
#define TINYTORCH_UTILS_UTILS_H_

#include <cstdlib>
#include <iostream>
#include <vector>

namespace TinyTorch {

inline int get_product_over_vector(const std::vector<int> vec) {
    int res = 1;
    for (auto &i : vec) {
        res *= i;
    }
    return res;
}

#define ASSERT(condition) \
    do { \
        if (!(condition)) { \
            std::cerr << "Assertion failed: " << #condition \
                      << ", file " << __FILE__ \
                      << ", line " << __LINE__ << ".\n"; \
            std::abort(); \
        } \
    } while (0)

}

#endif  // TINYTORCH_UTILS_UTILS_H_