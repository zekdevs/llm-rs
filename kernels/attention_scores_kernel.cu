extern "C" __global__ void attention_scores_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key,
    float* __restrict__ scores,
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

    int head_offset = ((batch * num_heads) + head) * seq_len * head_dim;

    const float* query_base = query + head_offset + query_pos * head_dim;
    const float* key_base = key + head_offset + key_pos * head_dim;

    float sum = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
        sum += query_base[d] * key_base[d];
    }

    scores[idx] = sum;
}
