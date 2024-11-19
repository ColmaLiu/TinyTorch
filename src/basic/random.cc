#include "basic/random.h"

#include <random>

namespace TinyTorch {

std::random_device rd;

std::mt19937_64 gen(rd());

}