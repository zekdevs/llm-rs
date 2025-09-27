#include <cuda_runtime.h>
#include <float.h>

extern "C" __global__ void max_reduce_kernel(const float* __restrict__ input,
                                               float* __restrict__ output,
                                               int rows,
                                               int cols) {
    extern __shared__ float shared[];
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= rows) {
        return;
    }

    float thread_max = -FLT_MAX;
    for (int col = tid; col < cols; col += blockDim.x) {
        float value = input[row * cols + col];
        thread_max = fmaxf(thread_max, value);
    }

    shared[tid] = thread_max;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[row] = shared[0];
    }
}
