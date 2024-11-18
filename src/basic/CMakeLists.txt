add_library(tinytorch_basic STATIC
    device.cc
    mem.cu
)

target_link_libraries(tinytorch_basic PUBLIC CUDA::cudart)