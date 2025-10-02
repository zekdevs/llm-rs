#include <cuda_runtime.h>

extern "C" __global__ void relu_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    float* __restrict__ grad_input,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) {
        return;
    }

    grad_input[idx] = input[idx] > 0.0f ? grad_output[idx] : 0.0f;
}
