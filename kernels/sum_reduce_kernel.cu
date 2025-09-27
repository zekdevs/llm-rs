#include <cuda_runtime.h>

extern "C" __global__ void sum_reduce_kernel(const float* __restrict__ input,
                                               float* __restrict__ output,
                                               int rows,
                                               int cols) {
    extern __shared__ float shared[];
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= rows) {
        return;
    }

    float thread_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        thread_sum += input[row * cols + col];
    }

    shared[tid] = thread_sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[row] = shared[0];
    }
}
