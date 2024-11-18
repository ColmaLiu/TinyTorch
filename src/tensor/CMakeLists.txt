add_library(tinytorch_tensor STATIC
    tensor.cu
)

target_link_libraries(tinytorch_tensor PUBLIC tinytorch_basic tinytorch_op CUDA::cudart)