#include <cuda_runtime.h>

extern "C" __global__ void embedding_backward_kernel(
    const float* __restrict__ grad_output,
    const int* __restrict__ indices,
    float* __restrict__ grad_weight,
    int batch_size,
    int seq_len,
    int embed_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * seq_len * embed_dim;

    if (idx >= total) {
        return;
    }

    int d = idx % embed_dim;
    int position = (idx / embed_dim) % seq_len;
    int batch = idx / (embed_dim * seq_len);

    int token_index = indices[batch * seq_len + position];
    if (token_index < 0) {
        return;
    }

    atomicAdd(grad_weight + token_index * embed_dim + d, grad_output[idx]);
}
