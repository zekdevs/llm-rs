#include <cuda_runtime.h>

extern "C" __global__ void attention_scores_backward_kernel(
    const float* __restrict__ grad_scores,
    const float* __restrict__ query,
    const float* __restrict__ key,
    float* __restrict__ grad_query,
    float* __restrict__ grad_key,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_heads * seq_len * seq_len;

    if (idx >= total) {
        return;
    }

    int key_pos = idx % seq_len;
    int query_pos = (idx / seq_len) % seq_len;
    int head = (idx / (seq_len * seq_len)) % num_heads;
    int batch = idx / (seq_len * seq_len * num_heads);

    int bh_index = batch * num_heads + head;
    int head_offset = bh_index * seq_len * head_dim;

    const float* query_vec = query + head_offset + query_pos * head_dim;
    const float* key_vec = key + head_offset + key_pos * head_dim;
    float* grad_query_vec = grad_query + head_offset + query_pos * head_dim;
    float* grad_key_vec = grad_key + head_offset + key_pos * head_dim;

    float grad = grad_scores[idx];

    for (int d = 0; d < head_dim; ++d) {
        atomicAdd(grad_query_vec + d, grad * key_vec[d]);
        atomicAdd(grad_key_vec + d, grad * query_vec[d]);
    }
}
