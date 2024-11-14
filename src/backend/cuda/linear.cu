#include "backend/cuda/linear.cuh"

#include <cublas_v2.h>

#include "backend/cuda/utils.cuh"

namespace TinyTorch::Backend::CUDA {

void linear_forward(float *input, float *output, float *weight, float *bias, int batchsize,
                    int in_features, int out_features) {
    gemm(CUBLAS_OP_N, CUBLAS_OP_N, batchsize, out_features, in_features, input, weight, output);
    mav1(bias, output, batchsize, out_features);
}

void linear_backward(float *input, float *output, float *weight, float *bias, int batchsize,
                     int in_features, int out_features, float *grad_output, float *grad_input,
                     float *grad_weight, float *grad_bias) {
    gemm(CUBLAS_OP_N, CUBLAS_OP_T, batchsize, in_features, out_features, grad_output, weight, grad_input);
    gemm(CUBLAS_OP_T, CUBLAS_OP_N, in_features, out_features, batchsize, input, grad_output, grad_weight);
    sumcol(grad_output, grad_bias, batchsize, out_features);
}

}