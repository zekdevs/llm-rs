extern "C" __global__ void matmul_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int m,
    int n,
    int k
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= m || col >= n) {
        return;
    }

    float sum = 0.0f;
    for (int e = 0; e < k; ++e) {
        sum += a[row * k + e] * b[e * n + col];
    }

    c[row * n + col] = sum;
}
