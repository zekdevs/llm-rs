#include <cuda_runtime.h>

extern "C" __global__ void embedding_lookup_kernel(
    const float* __restrict__ embeddings,
    const int* __restrict__ indices,
    float* __restrict__ output,
    int embed_dim,
    int seq_len,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * seq_len * embed_dim;

    if (idx >= total) {
        return;
    }

    int d = idx % embed_dim;
    int t = (idx / embed_dim) % seq_len;
    int b = idx / (embed_dim * seq_len);

    int token_index = indices[b * seq_len + t];
    output[idx] = embeddings[token_index * embed_dim + d];
}
