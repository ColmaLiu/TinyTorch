#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cassert>

typedef float scalar_t;

const int kCudaThreadsNum = 512;
inline int CudaGetBlocks(const int N)
{
    return (N + kCudaThreadsNum - 1) / kCudaThreadsNum;
}
#define CUDA_KERNEL_LOOP(n) \
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

__global__ void init_gpu(scalar_t* data, int size_)
{
    CUDA_KERNEL_LOOP(size_)
    {
        data[i] = 0.0f;
    }
}

__global__ void relu_forward_gpu(scalar_t* in, scalar_t* out, int n)
{
    CUDA_KERNEL_LOOP(n)
    {
        out[i] = in[i] > 0 ? in[i] : 0;
    }
}

__global__ void relu_backward_gpu(scalar_t* out_grad, scalar_t* in, scalar_t* in_grad, int n)
{
    CUDA_KERNEL_LOOP(n)
    {
        in_grad[i] = in[i] > 0 ? out_grad[i] : 0;
    }
}

__global__ void sigmoid_forward_gpu(scalar_t* in, scalar_t* out, int n)
{
    CUDA_KERNEL_LOOP(n)
    {
        out[i] = 1.0f / (1.0f + expf(-in[i]));
    }
}

__global__ void sigmoid_backward_gpu(scalar_t* out_grad, scalar_t* in, scalar_t* in_grad, int n)
{
    CUDA_KERNEL_LOOP(n)
    {
        scalar_t out = 1.0f / (1.0f + expf(-in[i]));
        in_grad[i] = out_grad[i] * out * (1 - out);
    }
}

class Tensor
{
public:
    std::vector<int> shape;
    bool host;
    size_t size;
    scalar_t* data;
    scalar_t* allocate_cpu()
    {
        scalar_t* data_ = new scalar_t[size];
        return data_;
    }
    scalar_t* allocate_gpu()
    {
        scalar_t* data_ = nullptr;
        cudaMalloc(&data_, size * sizeof(scalar_t));
        return data_;
    }
    Tensor(std::vector<int>& shape, bool host=true)
    : shape(shape), host(host), data(nullptr)
    {
        size = 1;
        for(int dim : shape)
        {
            size *= dim;
        }
        if(host)
        {
            data = allocate_cpu();
            memset(data, 0, size * sizeof(scalar_t));
        }
        else
        {
            data = allocate_gpu();
            init_gpu<<<CudaGetBlocks(size), kCudaThreadsNum>>>(data, size);
        }
    }
    ~Tensor()
    {
        if(data)
        {
            if(host)
            {
                delete[] data;
            }
            else
            {
                cudaFree(data);
            }
        }
    }
    Tensor(Tensor&& other) noexcept
    : shape(other.shape), host(other.host), size(other.size), data(other.data)
    {
        other.data = nullptr;
    }
    Tensor& operator=(Tensor&& other) noexcept
    {
        if(this != &other)
        {
            if(this->data)
            {
                if(this->host)
                {
                    delete[] this->data;
                }
                else
                {
                    cudaFree(this->data);
                }
            }
            this->data = other.data;
            other.data = nullptr;
            this->shape = other.shape;
            this->host = other.host;
            this->size = other.size;
        }
        return *this;
    }
    Tensor(const Tensor& other)
    : shape(other.shape), host(other.host), size(other.size)
    {
        if(this->host)
        {
            this->data = allocate_cpu();
            cudaMemcpy(this->data, other.data, this->size * sizeof(scalar_t), cudaMemcpyHostToHost);
        }
        else
        {
            this->data = allocate_gpu();
            cudaMemcpy(this->data, other.data, this->size * sizeof(scalar_t), cudaMemcpyDeviceToDevice);
        }
    }
    Tensor& operator=(const Tensor& other)
    {
        if(this != &other)
        {
            if(this->data)
            {
                if(this->host)
                {
                    delete[] this->data;
                }
                else
                {
                    cudaFree(this->data);
                }
            }
            this->shape = other.shape;
            this->host = other.host;
            this->size = other.size;
            if(this->host)
            {
                this->data = allocate_cpu();
                cudaMemcpy(this->data, other.data, this->size * sizeof(scalar_t), cudaMemcpyHostToHost);
            }
            else
            {
                this->data = allocate_gpu();
                cudaMemcpy(this->data, other.data, this->size * sizeof(scalar_t), cudaMemcpyDeviceToDevice);
            }
        }
        return *this;
    }
    Tensor cpu()
    {
        Tensor tensor(this->shape, true);
        if(this->host)
        {
            cudaMemcpy(tensor.data, this->data, this->size * sizeof(scalar_t), cudaMemcpyHostToHost);
        }
        else
        {
            cudaMemcpy(tensor.data, this->data, this->size * sizeof(scalar_t), cudaMemcpyDeviceToHost);
        }
        return tensor;
    }
    Tensor gpu()
    {
        Tensor tensor(this->shape, false);
        if(this->host)
        {
            cudaMemcpy(tensor.data, this->data, this->size * sizeof(scalar_t), cudaMemcpyHostToDevice);
        }
        else
        {
            cudaMemcpy(tensor.data, this->data, this->size * sizeof(scalar_t), cudaMemcpyDeviceToDevice);
        }
        return tensor;
    }
    void to_cpu()
    {
        if(!host)
        {
            scalar_t* data_ = allocate_cpu();
            cudaMemcpy(data_, data, size * sizeof(scalar_t), cudaMemcpyDeviceToHost);
            cudaFree(data);
            data = data_;
            host = true;
        }
    }
    void to_gpu()
    {
        if(host)
        {
            scalar_t* data_ = allocate_gpu();
            cudaMemcpy(data_, data, size * sizeof(scalar_t), cudaMemcpyHostToDevice);
            delete[] data;
            data = data_;
            host = false;
        }
    }
};

void relu_forward_cpu(scalar_t* in, scalar_t* out, int n)
{
    for(int i = 0; i < n; ++i)
    {
        out[i] = in[i] > 0 ? in[i] : 0;
    }
}

void relu_backward_cpu(scalar_t* out_grad, scalar_t* in, scalar_t* in_grad, int n)
{
    for(int i = 0; i < n; ++i)
    {
        in_grad[i] = in[i] > 0 ? out_grad[i] : 0;
    }
}

void sigmoid_forward_cpu(scalar_t* in, scalar_t* out, int n)
{
    for(int i = 0; i < n; ++i)
    {
        out[i] = 1.0f / (1.0f + expf(-in[i]));
    }
}

void sigmoid_backward_cpu(scalar_t* out_grad, scalar_t* in, scalar_t* in_grad, int n)
{
    for(int i = 0; i < n; ++i)
    {
        scalar_t out = 1.0f / (1.0f + expf(-in[i]));
        in_grad[i] = out_grad[i] * out * (1 - out);
    }
}

void relu_forward(Tensor* in, Tensor* out)
{
    assert(in->shape == out->shape);
    assert(in->host == out-> host);
    assert(in->size == out->size);
    if(out->host)
    {
        relu_forward_cpu(in->data, out->data, out->size);
    }
    else
    {
        relu_forward_gpu<<<CudaGetBlocks(out->size), kCudaThreadsNum>>>(in->data, out->data, out->size);
    }
}
void relu_backward(Tensor* out_grad, Tensor* in, Tensor* in_grad)
{
    assert(in_grad->shape == out_grad->shape && in_grad->shape == in->shape);
    assert(in_grad->host == out_grad->host && in_grad->host == in->host);
    assert(in_grad->size == out_grad->size && in_grad->size == in->size);
    if(in_grad->host)
    {
        relu_backward_cpu(out_grad->data, in->data, in_grad->data, in_grad->size);
    }
    else
    {
        relu_backward_gpu<<<CudaGetBlocks(in_grad->size), kCudaThreadsNum>>>(out_grad->data, in->data, in_grad->data, in_grad->size);
    }
}
void sigmoid_forward(Tensor* in, Tensor* out)
{
    assert(in->shape == out->shape);
    assert(in->host == out-> host);
    assert(in->size == out->size);
    if(out->host)
    {
        sigmoid_forward_cpu(in->data, out->data, out->size);
    }
    else
    {
        sigmoid_forward_gpu<<<CudaGetBlocks(out->size), kCudaThreadsNum>>>(in->data, out->data, out->size);
    }
}
void sigmoid_backward(Tensor* out_grad, Tensor* in, Tensor* in_grad)
{
    assert(in_grad->shape == out_grad->shape && in_grad->shape == in->shape);
    assert(in_grad->host == out_grad->host && in_grad->host == in->host);
    assert(in_grad->size == out_grad->size && in_grad->size == in->size);
    if(in_grad->host)
    {
        sigmoid_backward_cpu(out_grad->data, in->data, in_grad->data, in_grad->size);
    }
    else
    {
        sigmoid_backward_gpu<<<CudaGetBlocks(in_grad->size), kCudaThreadsNum>>>(out_grad->data, in->data, in_grad->data, in_grad->size);
    }
}
