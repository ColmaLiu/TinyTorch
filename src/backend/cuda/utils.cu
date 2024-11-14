#include "backend/cuda/utils.cuh"

#include <cublas_v2.h>

namespace TinyTorch::Backend::CUDA {

// C(m, n) = alp * op(A)(m, k) * op(B)(k, n) + bet * C(m, n)
void gemm(cublasOperation_t OP_A, cublasOperation_t OP_B, const int m, const int n, const int k,
              const float *A, const float *B, float *C, const float alp = 1.0f, const float bet = 0.0f) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    int lda = (OP_A == CUBLAS_OP_N) ? k : m;
    int ldb = (OP_B == CUBLAS_OP_N) ? n : k;
    int ldc = n;
    cublasSgemm(handle, OP_B, OP_A, n, m, k, &alp, B, ldb, A, lda, &bet, C, ldc);
    cublasDestroy(handle);
}

// Y(m, n) += X(n, )
void mav1(const float *X, float *Y, const int m, const int n) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alp = 1.0f;
    for (int i = 0; i < m; i++) {
      cublasSaxpy(handle, n, &alp, X, 1, Y + i * n, 1);
    }
    cublasDestroy(handle);
}

// Y(m, n) += X(m, )
void mav2(const float *X, float *Y, const int m, const int n) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alp = 1.0f;
    for (int i = 0; i < n; i++) {
      cublasSaxpy(handle, m, &alp, X, 1, Y + i, n);
    }
    cublasDestroy(handle);
}

// Y(n, ) = sum2d(X(m, n))
void sumcol(const float *X, float *Y, const int m, const int n) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alp = 1.0f;
    for (int i = 0; i < n; i++) {
        cublasSaxpy(handle, m, &alp, X + i, n, Y + i, 0);
    }
    cublasDestroy(handle);
}

__global__ void im2col_kernel(const int n, const float *data_im, const int height, const int width, const int ksize,
                                  const int pad, const int stride, const int height_col, const int width_col,
                                  float *data_col) {
    CUDA_KERNEL_LOOP(index, n) {
        int w_out = index % width_col;  //out_dada_W_th
        int h_index = index / width_col;//in_data_ch_th * out_data.H_th
        int h_out = h_index % height_col;//out_data.H_th
        int channel_in = h_index / height_col;//in_data.ch_th
        int channel_out = channel_in * ksize * ksize;//out_data.length
        int h_in = h_out * stride - pad;//(h_in, w_in) is the top-left point, where facing its counterpart in kernel
        int w_in = w_out * stride - pad;
        float* data_col_ptr = data_col;
        data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
        const float* data_im_ptr = data_im;
        data_im_ptr += (channel_in * height + h_in) * width + w_in;
        for (int i = 0; i < ksize; i++) {
            for (int j = 0; j < ksize; ++j) {
                int h = h_in + i;
                int w = w_in + j;
                *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
                    data_im_ptr[i * width + j] : 0.0f;
                data_col_ptr += height_col * width_col;
            }
        }
    }
}

// im(c, h, w) => col(c, k, k, hc, wc)
void im2col(const float *im, const int channels, const int height, const int width, const int ksize,
                const int pad, const int stride, float *col) {
    const int height_col = (height + 2 * pad - ksize) / stride + 1;
    const int width_col = (width + 2 * pad - ksize) / stride + 1;
    const int n = channels * height_col * width_col;
    im2col_kernel<<<CudaGetBlocks(n), kCudaThreadsNum>>>(n, im, height, width, ksize, pad, stride,
                                                             height_col, width_col, col);
}

__global__ void col2im_kernel(const int n, float *data_im, const int height, const int width, const int ksize,
                                  const int pad, const int stride, const int height_col, const int width_col,
                                  const float *data_col) {
    CUDA_KERNEL_LOOP(index, n) {
        int w_out = index % width;
        int h_index = index / width;
        int h_out = h_index % height;
        int channel_out = h_index / height;
        int channel_in = channel_out * ksize * ksize;
        const float *data_col_ptr = data_col;
        data_col_ptr += channel_in * height_col * width_col;
        float *data_im_ptr = data_im;
        data_im_ptr += (channel_out * height + h_out) * width + w_out;
        for (int i = 0; i < ksize; i++) {
            for (int j = 0; j < ksize; ++j) {
                int hps = h_out - i + pad;
                int wps = w_out - j + pad;
                int h = hps / stride;
                int w = wps / stride;
                *data_im_ptr += (hps % stride == 0 && wps % stride == 0 && h >= 0 && w >= 0 && h < height_col && w < width_col) ?
                    data_col_ptr[h * width_col + w] : 0.0f;
                data_col_ptr += height_col * width_col;
            }
        }
    }
}

// col(c, k, k, hc, wc) => im(c, h, w)
void col2im(float *im, const int channels, const int height, const int width, const int ksize,
                const int pad, const int stride, const float *col) {
    const int height_col = (height + 2 * pad - ksize) / stride + 1;
    const int width_col = (width + 2 * pad - ksize) / stride + 1;
    const int n = channels * height * width;
    col2im_kernel<<<CudaGetBlocks(n), kCudaThreadsNum>>>(n, im, height, width, ksize, pad, stride,
                                                             height_col, width_col, col);
}

__global__ void take_slice_kernel(const float *input, const float *index, float *output, const int m, const int n) {
    CUDA_KERNEL_LOOP(i, m) {
        output[i] = input[i * n + (int)index[i]];
    }
}

// output(m, ) = input(m, n)[range(m), index(m, )]
void take_slice(float *input, float *index, float *output, int m, int n) {
    take_slice_kernel<<<CudaGetBlocks(m), kCudaThreadsNum>>>(input, index, output, m, n);
}

__global__ void slice_sub1_kernel(float *x, const float *index, const int m, const int n) {
    CUDA_KERNEL_LOOP(i, m) {
        x[i * n + (int)index[i]] -= 1;
    }
}

// x(m, n)[range(m), index(m, )] -= 1
void slice_sub1(float *x, const float *index, const int m, const int n) {
    slice_sub1_kernel<<<CudaGetBlocks(m), kCudaThreadsNum>>>(x, index, m, n);
}

}