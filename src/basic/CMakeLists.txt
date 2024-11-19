add_library(tinytorch_basic STATIC
    device.cc
    mem.cu
    random.cc
)

target_link_libraries(tinytorch_basic PUBLIC CUDA::cudart)