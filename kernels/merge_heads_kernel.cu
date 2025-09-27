extern "C" __global__ void merge_heads_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_heads * seq_len * head_dim;

    if (idx >= total) {
        return;
    }

    int d = idx % head_dim;
    int s = (idx / head_dim) % seq_len;
    int h = (idx / (head_dim * seq_len)) % num_heads;
    int b = idx / (head_dim * seq_len * num_heads);

    int embed_dim = num_heads * head_dim;
    int output_index = ((b * seq_len + s) * embed_dim) + (h * head_dim + d);

    output[output_index] = input[idx];
}
