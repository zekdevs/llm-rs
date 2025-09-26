#include <cuda_runtime.h>

__global__ void add_scalar_kernel(const float* a, float scalar, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + scalar;
    }
}