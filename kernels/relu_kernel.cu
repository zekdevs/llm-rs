#include <cuda_runtime.h>

extern "C" __global__ void relu_kernel(const float* __restrict__ input,
                                         float* __restrict__ output,
                                         int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float value = input[idx];
        output[idx] = value > 0.0f ? value : 0.0f;
    }
}
