#include <cuda_runtime.h>

extern "C" __global__ void transpose2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cols || y >= rows) {
        return;
    }

    int input_idx = y * cols + x;
    int output_idx = x * rows + y;
    output[output_idx] = input[input_idx];
}
