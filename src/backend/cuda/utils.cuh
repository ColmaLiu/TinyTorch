#ifndef TINYTORCH_BACKEND_CUDA_UTILS_CUH_
#define TINYTORCH_BACKEND_CUDA_UTILS_CUH_

#include <cublas_v2.h>

namespace TinyTorch::Backend::CUDA {

constexpr int kCudaThreadsNum = 512;

inline int CudaGetBlocks(const int N) {
    return (N + kCudaThreadsNum - 1) / kCudaThreadsNum;
}

#define CUDA_KERNEL_LOOP(i, n) \
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

struct sub_scalar {
    const float a;
    inline sub_scalar(float a) : a(a) {}
    __host__ __device__
    constexpr float operator()(const float &x) const {
        return x - a;
    }
};

struct div_scalar {
    const float a;
    inline div_scalar(float a) : a(a) {}
    __host__ __device__
    constexpr float operator()(const float &x) const {
        return x / a;
    }
};

struct expo {
    inline expo() {}
    __host__ __device__
    constexpr float operator()(const float &x) const {
        return expf(x);
    }
};

struct neglog {
    inline neglog() {}
    __host__ __device__
    constexpr float operator()(const float &x) const {
        return -logf(x);
    }
};

// C(m, n) = alp * op(A)(m, k) * op(B)(k, n) + bet * C(m, n)
void gemm(cublasOperation_t OP_A, cublasOperation_t OP_B, const int m, const int n, const int k,
              const float *A, const float *B, float *C, const float alp = 1.0f, const float bet = 0.0f);

// Y(m, n) += X(n, )
void mav1(const float *X, float *Y, const int m, const int n);

// Y(m, n) += X(m, )
void mav2(const float *X, float *Y, const int m, const int n);

// Y(n, ) = sum2d(X(m, n))
void sumcol(const float *X, float *Y, const int m, const int n);

// im(c, h, w) => col(c, k, k, hc, wc)
void im2col(const float *im, const int channels, const int height, const int width, const int ksize,
                const int pad, const int stride, float *col);

// col(c, k, k, hc, wc) => im(c, h, w)
void col2im(float *im, const int channels, const int height, const int width, const int ksize,
                const int pad, const int stride, const float *col);

}

// output(m, ) = input(m, n)[range(m), index(m, )]
void take_slice(const float *input, const float *index, float *output, const int m, const int n);

// x(m, n)[range(m), index(m, )] -= 1
void slice_sub1(float *x, const float *index, const int m, const int n);

#endif  // TINYTORCH_BACKEND_CUDA_UTILS_CUH_