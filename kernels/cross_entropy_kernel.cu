#include <cuda_runtime.h>

extern "C" __global__ void cross_entropy_backward_kernel(
    const float* __restrict__ probs,
    const unsigned int* __restrict__ targets,
    float* __restrict__ grad_logits,
    float inv_batch,
    int vocab_size,
    int batch_seq
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_seq * vocab_size;
    if (idx >= total) {
        return;
    }

    int row = idx / vocab_size;
    int col = idx % vocab_size;
    unsigned int target = targets[row];

    float grad = probs[idx];
    if (col == static_cast<int>(target)) {
        grad -= 1.0f;
    }
    grad_logits[idx] = grad * inv_batch;
}

extern "C" __global__ void gather_target_prob_kernel(
    const float* __restrict__ probs,
    const unsigned int* __restrict__ targets,
    float* __restrict__ target_probs,
    int vocab_size,
    int batch_seq
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_seq) {
        return;
    }

    unsigned int target = targets[idx];
    if (target >= static_cast<unsigned int>(vocab_size)) {
        target = vocab_size - 1;
    }

    target_probs[idx] = probs[idx * vocab_size + target];
}
