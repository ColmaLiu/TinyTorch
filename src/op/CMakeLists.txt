add_library(tinytorch_op STATIC
    conv2d.cc
    cross_entropy.cc
    linear.cc
    max_pooling2d.cc
    relu.cc
    reshape.cc
    sigmoid.cc
    tensor_binary_op.cc
    tensor_eq.cc
    tensor_scalar_op.cc
    tensor_unary_op.cc
)

target_link_libraries(tinytorch_op PUBLIC
    tinytorch_tensor
    tinytorch_backend_cuda
)