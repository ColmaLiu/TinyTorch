set(PYMODULE_NAME "tinytorch")
pybind11_add_module(${PYMODULE_NAME} MODULE
    basic.cc
    op.cc
    pybind.cc
    tensor.cc
)

target_link_libraries(${PYMODULE_NAME} PRIVATE
    tinytorch_backend_cuda
    tinytorch_basic
    tinytorch_op
    tinytorch_tensor
)

target_compile_options(${PYMODULE_NAME} PRIVATE
    -flto=auto	# Supress the warning "lto-wrapper: warning: using serial compilation of 2 LTRANS jobs"
)