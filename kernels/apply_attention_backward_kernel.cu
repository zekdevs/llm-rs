#include <cuda_runtime.h>

extern "C" __global__ void apply_attention_backward_kernel(
    const float* __restrict__ grad_context,
    const float* __restrict__ attention,
    const float* __restrict__ value,
    float* __restrict__ grad_attention,
    float* __restrict__ grad_value,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_heads * seq_len * head_dim;

    if (idx >= total) {
        return;
    }

    int d = idx % head_dim;
    int q = (idx / head_dim) % seq_len;
    int h = (idx / (head_dim * seq_len)) % num_heads;
    int b = idx / (head_dim * seq_len * num_heads);

    int bh_index = b * num_heads + h;
    int value_offset = bh_index * seq_len * head_dim;
    int attention_offset = (bh_index * seq_len + q) * seq_len;

    float grad = grad_context[idx];

    const float* value_block = value + value_offset;
    const float* attention_row = attention + attention_offset;
    float* grad_attention_row = grad_attention + attention_offset;
    float* grad_value_block = grad_value + value_offset;

    for (int k = 0; k < seq_len; ++k) {
        float attn = attention_row[k];
        float val = value_block[k * head_dim + d];

        atomicAdd(grad_attention_row + k, grad * val);
        atomicAdd(grad_value_block + k * head_dim + d, grad * attn);
    }
}
