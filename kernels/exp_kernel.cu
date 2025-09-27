#include <cuda_runtime.h>
#include <math_constants.h>

extern "C" __global__ void exp_kernel(const float* __restrict__ input,
                                        float* __restrict__ output,
                                        int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = expf(input[idx]);
    }
}
