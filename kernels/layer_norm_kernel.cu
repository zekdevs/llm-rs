#include <cuda_runtime.h>

extern "C" __global__ void layer_norm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int rows,
    int cols,
    float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) {
        return;
    }

    extern __shared__ float shared[];
    float* shared_sum = shared;
    float* shared_sq_sum = shared + blockDim.x;

    float sum = 0.0f;
    float sq_sum = 0.0f;

    for (int col = tid; col < cols; col += blockDim.x) {
        float value = input[row * cols + col];
        sum += value;
        sq_sum += value * value;
    }

    shared_sum[tid] = sum;
    shared_sq_sum[tid] = sq_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
            shared_sq_sum[tid] += shared_sq_sum[tid + stride];
        }
        __syncthreads();
    }

    float mean = shared_sum[0] / static_cast<float>(cols);
    float variance = shared_sq_sum[0] / static_cast<float>(cols) - mean * mean;
    if (variance < 0.0f) {
        variance = 0.0f;
    }
    float inv_std = rsqrtf(variance + eps);

    if (tid == 0) {
        shared_sum[0] = mean;
        shared_sq_sum[0] = inv_std;
    }
    __syncthreads();

    mean = shared_sum[0];
    inv_std = shared_sq_sum[0];

    for (int col = tid; col < cols; col += blockDim.x) {
        float value = input[row * cols + col];
        float normalized = (value - mean) * inv_std;
        float scaled = normalized * gamma[col] + beta[col];
        output[row * cols + col] = scaled;
    }
}
