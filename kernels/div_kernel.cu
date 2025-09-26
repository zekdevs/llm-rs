#include <cuda_runtime.h>

__device__ __forceinline__ size_t compute_offset(int index, const size_t* shape, const size_t* strides, int rank) {
    if (rank == 0) {
        return 0;
    }

    size_t linear = static_cast<size_t>(index);
    size_t offset = 0;
    for (int axis = rank - 1; axis >= 0; --axis) {
        size_t dim = shape[axis];
        size_t coord = linear % dim;
        linear /= dim;
        offset += coord * strides[axis];
    }
    return offset;
}

extern "C" __global__ void div_kernel(
    const float* a,
    const float* b,
    float* c,
    int n,
    const size_t* shape,
    const size_t* strides_a,
    const size_t* strides_b,
    int rank
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        size_t offset_a = compute_offset(idx, shape, strides_a, rank);
        size_t offset_b = compute_offset(idx, shape, strides_b, rank);
        c[idx] = a[offset_a] / b[offset_b];
    }
}