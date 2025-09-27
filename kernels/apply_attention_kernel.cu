extern "C" __global__ void apply_attention_kernel(
    const float* __restrict__ attention,
    const float* __restrict__ value,
    float* __restrict__ output,
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
    int query_pos = (idx / head_dim) % seq_len;
    int head = (idx / (head_dim * seq_len)) % num_heads;
    int batch = idx / (head_dim * seq_len * num_heads);

    int attn_base_index = (((batch * num_heads) + head) * seq_len + query_pos) * seq_len;
    const float* attn_base = attention + attn_base_index;

    int value_base_index = ((batch * num_heads) + head) * seq_len * head_dim;
    const float* value_base = value + value_base_index;

    float sum = 0.0f;
    for (int key_pos = 0; key_pos < seq_len; ++key_pos) {
        float weight = attn_base[key_pos];
        float v = value_base[key_pos * head_dim + d];
        sum += weight * v;
    }

    output[idx] = sum;
}
