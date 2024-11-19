add_library(tinytorch_backend_cuda STATIC
    conv2d.cu
    cross_entropy.cu
    linear.cu
    max_pool2d.cu
    relu.cu
    sigmoid.cu
    tensor_binary_op.cu
    tensor_close.cu
    tensor_eq.cu
    tensor_scalar_op.cu
    tensor_unary_op.cu
    utils.cu
)

target_link_libraries(tinytorch_backend_cuda PUBLIC CUDA::cudart CUDA::cublas)