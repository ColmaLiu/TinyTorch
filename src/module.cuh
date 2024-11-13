#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>

#include <iostream>

const int kCudaThreadsNum = 512;
inline int CudaGetBlocks(const int N) {
    return (N + kCudaThreadsNum - 1) / kCudaThreadsNum;
}
#define CUDA_KERNEL_LOOP(i, n) \
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

// C(m, n) = op(A)(m, k) * op(B)(k, n)
void gemm_gpu(cublasOperation_t OP_A, cublasOperation_t OP_B, const int m, const int n, const int k,
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
void mav1_gpu(const float *X, float *Y, const int m, const int n) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alp = 1.0f;
    for (int i = 0; i < m; i++) {
      cublasSaxpy(handle, n, &alp, X, 1, Y + i * n, 1);
    }
    cublasDestroy(handle);
}

// Y(m, n) += X(m, )
void mav2_gpu(const float *X, float *Y, const int m, const int n) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alp = 1.0f;
    for (int i = 0; i < n; i++) {
      cublasSaxpy(handle, m, &alp, X, 1, Y + i, n);
    }
    cublasDestroy(handle);
}

// Y(n, ) = sum2d(X(m, n))
void sumcol_gpu(const float *X, float *Y, const int m, const int n) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alp = 1.0f;
    for (int i = 0; i < n; i++) {
        cublasSaxpy(handle, m, &alp, X + i, n, Y + i, 0);
    }
    cublasDestroy(handle);
}

__global__ void im2col_gpu_kernel(const int n, const float *data_im, const int height, const int width, const int ksize,
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
void im2col_gpu(const float *im, const int channels, const int height, const int width, const int ksize,
                const int pad, const int stride, float *col) {
    const int height_col = (height + 2 * pad - ksize) / stride + 1;
    const int width_col = (width + 2 * pad - ksize) / stride + 1;
    const int n = channels * height_col * width_col;
    im2col_gpu_kernel<<<CudaGetBlocks(n), kCudaThreadsNum>>>(n, im, height, width, ksize, pad, stride,
                                                             height_col, width_col, col);
}

__global__ void col2im_gpu_kernel(const int n, float *data_im, const int height, const int width, const int ksize,
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
void col2im_gpu(float *im, const int channels, const int height, const int width, const int ksize,
                const int pad, const int stride, const float *col) {
    const int height_col = (height + 2 * pad - ksize) / stride + 1;
    const int width_col = (width + 2 * pad - ksize) / stride + 1;
    const int n = channels * height * width;
    col2im_gpu_kernel<<<CudaGetBlocks(n), kCudaThreadsNum>>>(n, im, height, width, ksize, pad, stride,
                                                             height_col, width_col, col);
}

__global__ void forward_maxpool_kernel(const int n, const float *input, float *output, float *mask,
                                       const int height_in, const int width_in,
                                       const int height_out, const int width_out,
                                       const int ksize, const int pad, const int stride) {
    CUDA_KERNEL_LOOP(index, n) {
        int w_out = index % width_out;
        int h_index = index / width_out;
        int h_out = h_index % height_out;
        int ch = h_index / height_out;
        int h_in = h_out * stride - pad;
        int w_in = w_out * stride - pad;
        int offset = (ch * height_in + h_in) * width_in + w_in;
        const float *data_ptr = input;
        data_ptr += offset;
        int maxi = 0, maxj = 0;
        for (int i = 0; i < ksize; i++) {
            for (int j = 0; j < ksize; ++j) {
                int h = h_in + i;
                int w = w_in + j;
                if (h >= 0 && w >= 0 && h < height_in && w < width_in
                    && data_ptr[i * width_in + j] > data_ptr[maxi * width_in + maxj]) {
                    maxi = i;
                    maxj = j;
                }
            }
        }
        output[index] = data_ptr[maxi * width_in + maxj];
        mask[offset + maxi * width_in + maxj] = 1.0f;
    }
}

__global__ void backward_maxpool_kernel(const int n, const float *input, const float *output, const float *mask,
                                       const int height_in, const int width_in,
                                       const int height_out, const int width_out,
                                       const int ksize, const int pad, const int stride,
                                       float *grad_output, float *grad_input) {
    CUDA_KERNEL_LOOP(index, n) {
        int w_out = index % width_out;
        int h_index = index / width_out;
        int h_out = h_index % height_out;
        int ch = h_index / height_out;
        int h_in = h_out * stride - pad;
        int w_in = w_out * stride - pad;
        int offset = (ch * height_in + h_in) * width_in + w_in;
        const float *mask_ptr = mask;
        mask_ptr += offset;
        for (int i = 0; i < ksize; i++) {
            for (int j = 0; j < ksize; ++j) {
                int h = h_in + i;
                int w = w_in + j;
                if (h >= 0 && w >= 0 && h < height_in && w < width_in
                    && mask_ptr[i * width_in + j] == 1.0f) {
                    grad_input[offset + i * width_in + j] = grad_output[index];
                    break;
                }
            }
        }
    }
}

struct sub_scalar {
    const float a;
    sub_scalar(float a) : a(a) {}
    __host__ __device__
    constexpr float operator()(const float &x) const {
        return x - a;
    }
};

struct div_scalar {
    const float a;
    div_scalar(float a) : a(a) {}
    __host__ __device__
    constexpr float operator()(const float &x) const {
        return x / a;
    }
};

struct expo {
    expo() {}
    __host__ __device__
    constexpr float operator()(const float &x) const {
        return expf(x);
    }
};

struct neglog {
    neglog() {}
    __host__ __device__
    constexpr float operator()(const float &x) const {
        return -logf(x);
    }
};

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

void matrix_init(float *A, int size) {
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());
    curandGenerateUniform(prng, A, size);
    curandDestroyGenerator(prng);
}

void forward_fc(float *input, float *output, float *weight, float *bias, int batchsize,
                int in_features, int out_features) {
    gemm_gpu(CUBLAS_OP_N, CUBLAS_OP_N, batchsize, out_features, in_features, input, weight, output);
    mav1_gpu(bias, output, batchsize, out_features);
}

void backward_fc(float *input, float *output, float *weight, float *bias, int batchsize,
                 int in_features, int out_features, float *grad_output, float *grad_input,
                 float *grad_weights, float *grad_bias) {
    gemm_gpu(CUBLAS_OP_N, CUBLAS_OP_T, batchsize, in_features, out_features, grad_output, weight, grad_input);
    gemm_gpu(CUBLAS_OP_T, CUBLAS_OP_N, in_features, out_features, batchsize, input, grad_output, grad_weights);
    sumcol_gpu(grad_output, grad_bias, batchsize, out_features);
}

void forward_conv(float *input, float *output, float *kernel, float *bias, int batchsize, int channels_in, int channels_out,
                  int height, int width, int ksize, int pad, int stride) {
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    thrust::device_vector<float> col(channels_in * ksize * ksize * height_col * width_col);
    for (int i = 0; i < batchsize; i++) {
        im2col_gpu(input + channels_in * height * width * i, channels_in, height, width, ksize, pad, stride,
                   thrust::raw_pointer_cast(col.data()));
        cudaDeviceSynchronize();
        gemm_gpu(CUBLAS_OP_N, CUBLAS_OP_N, channels_out, height_col * width_col, channels_in * ksize * ksize,
                 kernel, thrust::raw_pointer_cast(col.data()), output + channels_out * height_col * width_col * i);
        mav2_gpu(bias, output + channels_out * height_col * width_col * i, channels_out, height_col * width_col);
    }
}

void backward_conv(float *input, float *output, float *kernel, float *bias, int batchsize, int channels_in, int channels_out,
                   int height, int width, int ksize, int pad, int stride, float *grad_output, float *grad_input,
                   float *grad_kernel, float *grad_bias) {
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    thrust::device_vector<float> grad_col(channels_in * ksize * ksize * height_col * width_col);
    thrust::device_vector<float> col(channels_in * ksize * ksize * height_col * width_col);
    thrust::device_ptr<float> grad_in(grad_input);
    thrust::device_ptr<float> grad_out(grad_output);
    thrust::device_vector<float> grad_bias_(channels_out);
    // kernel   cout, cin, k, k
    // output   n, cout, hc, wc
    // col      cin, k, k, hc, wc
    // input    n, cin, h, w
    for (int i = 0; i < batchsize; i++) {
        gemm_gpu(CUBLAS_OP_T, CUBLAS_OP_N, channels_in * ksize * ksize, height_col * width_col, channels_out, kernel,
                 grad_output + channels_out * height_col * width_col * i, thrust::raw_pointer_cast(grad_col.data()));
        col2im_gpu(grad_input + channels_in * height * width * i, channels_in, height, width, ksize, pad, stride,
                   thrust::raw_pointer_cast(grad_col.data()));
        cudaDeviceSynchronize();
        // calc grad_kernel
        im2col_gpu(input + channels_in * height * width * i, channels_in, height, width, ksize, pad, stride,
                   thrust::raw_pointer_cast(col.data()));
        cudaDeviceSynchronize();
        gemm_gpu(CUBLAS_OP_N, CUBLAS_OP_T, channels_out, channels_in * ksize * ksize, height_col * width_col,
                 grad_output + channels_out * height_col * width_col * i, thrust::raw_pointer_cast(col.data()), grad_kernel, 1.0f, 1.0f);
        // calc grad_bias
        for (int j = 0; j < channels_out; j++) {
            grad_bias_[j] += thrust::reduce(grad_out + channels_out * height_col * width_col * i + height_col * width_col * j,
                           grad_out + channels_out * height_col * width_col * i + height_col * width_col * (j + 1),
                           0.0f, thrust::plus<float>());
        }
    }
    thrust::copy(grad_bias_.begin(), grad_bias_.end(), grad_bias);
}

void forward_maxpool(float *input, float *output, float *mask, int batchsize, int channels, int height, int width,
                     int ksize, int pad, int stride) {
    int height_out = (height + 2 * pad - ksize) / stride + 1;
    int width_out = (width + 2 * pad - ksize) / stride + 1;
    int n = batchsize * channels * height_out * width_out;
    forward_maxpool_kernel<<<CudaGetBlocks(n), kCudaThreadsNum>>>(n, input, output, mask, height, width,
                                                                  height_out, width_out, ksize, pad, stride);
}

void backward_maxpool(float *input, float *output, float *mask, int batchsize, int channels, int height, int width,
                     int ksize, int pad, int stride, float *grad_output, float *grad_input) {
    int height_out = (height + 2 * pad - ksize) / stride + 1;
    int width_out = (width + 2 * pad - ksize) / stride + 1;
    int n = batchsize * channels * height_out * width_out;
    backward_maxpool_kernel<<<CudaGetBlocks(n), kCudaThreadsNum>>>(n, input, output, mask, height, width,
                                                                   height_out, width_out, ksize, pad, stride,
                                                                   grad_output, grad_input);
}

void forward_softmax(float *input, float *output, int batchsize, int labels) {
    thrust::device_ptr<float> in(input);
    thrust::device_ptr<float> out(output);
    thrust::copy(in, in + batchsize * labels, out);
    for (int i = 0; i < batchsize; i++) {
        float max = thrust::reduce(out + i * labels, out + (i + 1) * labels, -FLT_MAX, thrust::maximum<float>());
        thrust::transform(out + i * labels, out + (i + 1) * labels, out + i * labels, sub_scalar(max));
        thrust::transform(out + i * labels, out + (i + 1) * labels, out + i * labels, expo());
        float sum = thrust::reduce(out + i * labels, out + (i + 1) * labels, 0.0f, thrust::plus<float>());
        thrust::transform(out + i * labels, out + (i + 1) * labels, out + i * labels, div_scalar(sum));
    }
}

void forward_celoss(float *input, float *target, float *loss, int batchsize, int labels) {
    thrust::device_vector<float> ll(batchsize);
    thrust::device_ptr<float> l(loss);
    take_slice(input, target, thrust::raw_pointer_cast(ll.data()), batchsize, labels);
    thrust::transform(ll.begin(), ll.end(), ll.begin(), neglog());
    *l = thrust::reduce(ll.begin(), ll.end(), 0.0f, thrust::plus<float>()) / batchsize;
}

void backward_celoss_with_softmax(float *input, float *prob, float *target, float *loss, int batchsize, int labels,
                                  float *grad_input) {
    thrust::device_ptr<float> p(prob);
    thrust::device_ptr<float> grad_in(grad_input);
    thrust::copy(p, p + batchsize * labels, grad_in);
    slice_sub1(grad_input, target, batchsize, labels);
    thrust::transform(grad_in, grad_in + batchsize * labels, grad_in, div_scalar(batchsize));
}
