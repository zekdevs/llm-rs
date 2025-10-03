use crate::kernel_cache;
use anyhow::{Context, Result, anyhow};
use cust::device::Device;
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use serde::{Deserialize, Serialize};
use std::convert::TryFrom;

use crate::tensor::Tensor;

pub trait Layer {
    fn forward(&self, input: &Tensor) -> Result<Tensor>;
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ActivationKind {
    Relu,
}

pub struct ActivationLayer {
    activation: ActivationKind,
}

impl ActivationLayer {
    pub fn new(activation: ActivationKind) -> Self {
        Self { activation }
    }
}

impl Layer for ActivationLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        match self.activation {
            ActivationKind::Relu => input.relu(),
        }
    }
}

pub struct SoftmaxLayer;

impl SoftmaxLayer {
    pub fn new() -> Self {
        Self
    }
}

impl Layer for SoftmaxLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if input.shape().len() != 2 {
            return Err(anyhow!(
                "SoftmaxLayer expects input shaped [batch, features], got {:?}",
                input.shape()
            ));
        }
        input.softmax()
    }
}

pub struct MultiHeadAttention {
    embed_dim: usize,
    num_heads: usize,
    head_dim: usize,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    w_o: Tensor,
    b_q: Option<Tensor>,
    b_k: Option<Tensor>,
    b_v: Option<Tensor>,
    b_o: Option<Tensor>,
}

pub struct MultiHeadAttentionCache {
    input_shape: Vec<usize>,
    batch_size: usize,
    seq_len: usize,
    flattened_input: Tensor,
    q_heads: Tensor,
    k_heads: Tensor,
    v_heads: Tensor,
    probs_4d: Tensor,
    probs_flat: Tensor,
    context_flat: Tensor,
}

pub struct MultiHeadAttentionGradients {
    pub w_q: Tensor,
    pub w_k: Tensor,
    pub w_v: Tensor,
    pub w_o: Tensor,
    pub b_q: Option<Tensor>,
    pub b_k: Option<Tensor>,
    pub b_v: Option<Tensor>,
    pub b_o: Option<Tensor>,
}

impl MultiHeadAttention {
    pub fn new(device: &Device, embed_dim: usize, num_heads: usize) -> Result<Self> {
        if num_heads == 0 {
            return Err(anyhow!("num_heads must be greater than zero"));
        }
        if embed_dim % num_heads != 0 {
            return Err(anyhow!(
                "embed_dim ({}) must be divisible by num_heads ({})",
                embed_dim,
                num_heads
            ));
        }

        let scale = 1.0f32 / (embed_dim as f32).sqrt();
        let w_q = Tensor::randn(vec![embed_dim, embed_dim], device)?;
        let w_q = w_q.mul_scalar(scale)?;
        let w_k = Tensor::randn(vec![embed_dim, embed_dim], device)?;
        let w_k = w_k.mul_scalar(scale)?;
        let w_v = Tensor::randn(vec![embed_dim, embed_dim], device)?;
        let w_v = w_v.mul_scalar(scale)?;
        let w_o = Tensor::randn(vec![embed_dim, embed_dim], device)?;
        let w_o = w_o.mul_scalar(scale)?;

        Self::assemble(
            embed_dim, num_heads, w_q, w_k, w_v, w_o, None, None, None, None,
        )
    }

    pub fn from_host(
        device: &Device,
        embed_dim: usize,
        num_heads: usize,
        w_q: Vec<f32>,
        w_k: Vec<f32>,
        w_v: Vec<f32>,
        w_o: Vec<f32>,
        b_q: Option<Vec<f32>>,
        b_k: Option<Vec<f32>>,
        b_v: Option<Vec<f32>>,
        b_o: Option<Vec<f32>>,
    ) -> Result<Self> {
        let w_q = Tensor::from_host(w_q, vec![embed_dim, embed_dim], device)?;
        let w_k = Tensor::from_host(w_k, vec![embed_dim, embed_dim], device)?;
        let w_v = Tensor::from_host(w_v, vec![embed_dim, embed_dim], device)?;
        let w_o = Tensor::from_host(w_o, vec![embed_dim, embed_dim], device)?;

        let b_q = match b_q {
            Some(data) => Some(Tensor::from_host(data, vec![1, embed_dim], device)?),
            None => None,
        };
        let b_k = match b_k {
            Some(data) => Some(Tensor::from_host(data, vec![1, embed_dim], device)?),
            None => None,
        };
        let b_v = match b_v {
            Some(data) => Some(Tensor::from_host(data, vec![1, embed_dim], device)?),
            None => None,
        };
        let b_o = match b_o {
            Some(data) => Some(Tensor::from_host(data, vec![1, embed_dim], device)?),
            None => None,
        };

        Self::assemble(embed_dim, num_heads, w_q, w_k, w_v, w_o, b_q, b_k, b_v, b_o)
    }

    fn assemble(
        embed_dim: usize,
        num_heads: usize,
        w_q: Tensor,
        w_k: Tensor,
        w_v: Tensor,
        w_o: Tensor,
        b_q: Option<Tensor>,
        b_k: Option<Tensor>,
        b_v: Option<Tensor>,
        b_o: Option<Tensor>,
    ) -> Result<Self> {
        if embed_dim % num_heads != 0 {
            return Err(anyhow!(
                "embed_dim ({}) must be divisible by num_heads ({})",
                embed_dim,
                num_heads
            ));
        }

        let head_dim = embed_dim / num_heads;
        let device_raw = w_q.device.as_raw();
        for tensor in [&w_k, &w_v, &w_o] {
            if tensor.device.as_raw() != device_raw {
                return Err(anyhow!("All weight tensors must live on the same device"));
            }
            if tensor.shape() != &[embed_dim, embed_dim] {
                return Err(anyhow!(
                    "Weights must have shape [{}, {}], found {:?}",
                    embed_dim,
                    embed_dim,
                    tensor.shape()
                ));
            }
        }

        for bias in [&b_q, &b_k, &b_v, &b_o] {
            if let Some(bias) = bias {
                if bias.device.as_raw() != device_raw {
                    return Err(anyhow!("All bias tensors must live on the same device"));
                }
                if bias.shape() != &[1, embed_dim] {
                    return Err(anyhow!(
                        "Biases must have shape [1, {}], found {:?}",
                        embed_dim,
                        bias.shape()
                    ));
                }
            }
        }

        Ok(Self {
            embed_dim,
            num_heads,
            head_dim,
            w_q,
            w_k,
            w_v,
            w_o,
            b_q,
            b_k,
            b_v,
            b_o,
        })
    }

    fn project(&self, input: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
        let mut projected = input.matmul(weight)?;
        if let Some(bias) = bias {
            projected = projected.add(bias)?;
        }
        Ok(projected)
    }

    fn split_heads(&self, tensor: &Tensor, batch_size: usize, seq_len: usize) -> Result<Tensor> {
        if tensor.shape().len() != 3
            || tensor.shape()[0] != batch_size
            || tensor.shape()[1] != seq_len
            || tensor.shape()[2] != self.embed_dim
        {
            return Err(anyhow!(
                "Expected tensor shape [{}, {}, {}], found {:?}",
                batch_size,
                seq_len,
                self.embed_dim,
                tensor.shape()
            ));
        }

        let output_shape = vec![batch_size, self.num_heads, seq_len, self.head_dim];
        let result = Tensor::new(output_shape, &tensor.device)?;
        let total = batch_size
            .checked_mul(self.num_heads)
            .and_then(|v| v.checked_mul(seq_len))
            .and_then(|v| v.checked_mul(self.head_dim))
            .ok_or_else(|| anyhow!("split_heads size overflow"))?;
        if total > i32::MAX as usize {
            return Err(anyhow!("split_heads tensor too large for kernel launch"));
        }

        let block_size = 256u32;
        let grid_size = ((total as u32) + block_size - 1) / block_size;
        let stream =
            Stream::new(StreamFlags::NON_BLOCKING, None).context("Failed to create CUDA stream")?;
        let module = kernel_cache::module(include_str!("split_heads_kernel.ptx"), "split_heads")?;
        let function = module
            .get_function("split_heads_kernel")
            .context("Kernel load failed for split_heads")?;

        unsafe {
            launch!(function<<<grid_size, block_size, 0, stream>>>(
                tensor.data.as_device_ptr(),
                result.data.as_device_ptr(),
                batch_size as i32,
                seq_len as i32,
                self.num_heads as i32,
                self.head_dim as i32
            ))?;
        }

        stream
            .synchronize()
            .context("Stream sync failed for split_heads")?;
        Ok(result)
    }

    fn merge_heads(&self, tensor: &Tensor, batch_size: usize, seq_len: usize) -> Result<Tensor> {
        if tensor.shape().len() != 4
            || tensor.shape()[0] != batch_size
            || tensor.shape()[1] != self.num_heads
            || tensor.shape()[2] != seq_len
            || tensor.shape()[3] != self.head_dim
        {
            return Err(anyhow!(
                "Expected tensor shape [{}, {}, {}, {}], found {:?}",
                batch_size,
                self.num_heads,
                seq_len,
                self.head_dim,
                tensor.shape()
            ));
        }

        let output_shape = vec![batch_size, seq_len, self.embed_dim];
        let result = Tensor::new(output_shape, &tensor.device)?;
        let total = batch_size
            .checked_mul(self.num_heads)
            .and_then(|v| v.checked_mul(seq_len))
            .and_then(|v| v.checked_mul(self.head_dim))
            .ok_or_else(|| anyhow!("merge_heads size overflow"))?;
        if total > i32::MAX as usize {
            return Err(anyhow!("merge_heads tensor too large for kernel launch"));
        }

        let block_size = 256u32;
        let grid_size = ((total as u32) + block_size - 1) / block_size;
        let stream =
            Stream::new(StreamFlags::NON_BLOCKING, None).context("Failed to create CUDA stream")?;
        let module = kernel_cache::module(include_str!("merge_heads_kernel.ptx"), "merge_heads")?;
        let function = module
            .get_function("merge_heads_kernel")
            .context("Kernel load failed for merge_heads")?;

        unsafe {
            launch!(function<<<grid_size, block_size, 0, stream>>>(
                tensor.data.as_device_ptr(),
                result.data.as_device_ptr(),
                batch_size as i32,
                seq_len as i32,
                self.num_heads as i32,
                self.head_dim as i32
            ))?;
        }

        stream
            .synchronize()
            .context("Stream sync failed for merge_heads")?;
        Ok(result)
    }

    fn compute_attention_scores(
        &self,
        query: &Tensor,
        key: &Tensor,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        for tensor in [query, key] {
            if tensor.shape().len() != 4
                || tensor.shape()[0] != batch_size
                || tensor.shape()[1] != self.num_heads
                || tensor.shape()[2] != seq_len
                || tensor.shape()[3] != self.head_dim
            {
                return Err(anyhow!(
                    "Expected tensor shape [{}, {}, {}, {}], found {:?}",
                    batch_size,
                    self.num_heads,
                    seq_len,
                    self.head_dim,
                    tensor.shape()
                ));
            }
        }

        let output_shape = vec![batch_size, self.num_heads, seq_len, seq_len];
        let result = Tensor::new(output_shape, &query.device)?;
        let total = batch_size
            .checked_mul(self.num_heads)
            .and_then(|v| v.checked_mul(seq_len))
            .and_then(|v| v.checked_mul(seq_len))
            .ok_or_else(|| anyhow!("attention score size overflow"))?;
        if total > i32::MAX as usize {
            return Err(anyhow!(
                "attention score tensor too large for kernel launch"
            ));
        }

        let block_size = 256u32;
        let grid_size = ((total as u32) + block_size - 1) / block_size;
        let stream =
            Stream::new(StreamFlags::NON_BLOCKING, None).context("Failed to create CUDA stream")?;
        let module = kernel_cache::module(
            include_str!("attention_scores_kernel.ptx"),
            "attention_scores",
        )?;
        let function = module
            .get_function("attention_scores_kernel")
            .context("Kernel load failed for attention_scores")?;

        unsafe {
            launch!(function<<<grid_size, block_size, 0, stream>>>(
                query.data.as_device_ptr(),
                key.data.as_device_ptr(),
                result.data.as_device_ptr(),
                batch_size as i32,
                self.num_heads as i32,
                seq_len as i32,
                self.head_dim as i32
            ))?;
        }

        stream
            .synchronize()
            .context("Stream sync failed for attention_scores")?;
        Ok(result)
    }

    fn apply_attention(
        &self,
        attention: &Tensor,
        value: &Tensor,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        if attention.shape().len() != 4
            || attention.shape()[0] != batch_size
            || attention.shape()[1] != self.num_heads
            || attention.shape()[2] != seq_len
            || attention.shape()[3] != seq_len
        {
            return Err(anyhow!(
                "Expected attention shape [{}, {}, {}, {}], found {:?}",
                batch_size,
                self.num_heads,
                seq_len,
                seq_len,
                attention.shape()
            ));
        }
        if value.shape().len() != 4
            || value.shape()[0] != batch_size
            || value.shape()[1] != self.num_heads
            || value.shape()[2] != seq_len
            || value.shape()[3] != self.head_dim
        {
            return Err(anyhow!(
                "Expected value shape [{}, {}, {}, {}], found {:?}",
                batch_size,
                self.num_heads,
                seq_len,
                self.head_dim,
                value.shape()
            ));
        }

        let output_shape = vec![batch_size, self.num_heads, seq_len, self.head_dim];
        let result = Tensor::new(output_shape, &attention.device)?;
        let total = batch_size
            .checked_mul(self.num_heads)
            .and_then(|v| v.checked_mul(seq_len))
            .and_then(|v| v.checked_mul(self.head_dim))
            .ok_or_else(|| anyhow!("apply_attention size overflow"))?;
        if total > i32::MAX as usize {
            return Err(anyhow!(
                "apply_attention tensor too large for kernel launch"
            ));
        }

        let block_size = 256u32;
        let grid_size = ((total as u32) + block_size - 1) / block_size;
        let stream =
            Stream::new(StreamFlags::NON_BLOCKING, None).context("Failed to create CUDA stream")?;
        let module = kernel_cache::module(
            include_str!("apply_attention_kernel.ptx"),
            "apply_attention",
        )?;
        let function = module
            .get_function("apply_attention_kernel")
            .context("Kernel load failed for apply_attention")?;

        unsafe {
            launch!(function<<<grid_size, block_size, 0, stream>>>(
                attention.data.as_device_ptr(),
                value.data.as_device_ptr(),
                result.data.as_device_ptr(),
                batch_size as i32,
                self.num_heads as i32,
                seq_len as i32,
                self.head_dim as i32
            ))?;
        }

        stream
            .synchronize()
            .context("Stream sync failed for apply_attention")?;
        Ok(result)
    }

    pub fn forward_with_cache(&self, input: &Tensor) -> Result<(Tensor, MultiHeadAttentionCache)> {
        if input.shape().len() != 3 {
            return Err(anyhow!(
                "MultiHeadAttention expects input shaped [batch, seq_len, embed_dim], got {:?}",
                input.shape()
            ));
        }

        let batch_size = input.shape()[0];
        let seq_len = input.shape()[1];
        let embed_dim = input.shape()[2];

        if embed_dim != self.embed_dim {
            return Err(anyhow!(
                "Input embed_dim ({}) does not match layer embed_dim ({})",
                embed_dim,
                self.embed_dim
            ));
        }

        if input.device.as_raw() != self.w_q.device.as_raw() {
            return Err(anyhow!(
                "Input tensor and layer weights must share the same device"
            ));
        }

        let input_shape = input.shape().to_vec();

        let flattened = input
            .clone()
            .reshape(vec![batch_size * seq_len, embed_dim])?;

        let q_linear = self.project(&flattened, &self.w_q, self.b_q.as_ref())?;
        let k_linear = self.project(&flattened, &self.w_k, self.b_k.as_ref())?;
        let v_linear = self.project(&flattened, &self.w_v, self.b_v.as_ref())?;

        let q_seq = q_linear
            .clone()
            .reshape(vec![batch_size, seq_len, embed_dim])?;
        let k_seq = k_linear
            .clone()
            .reshape(vec![batch_size, seq_len, embed_dim])?;
        let v_seq = v_linear
            .clone()
            .reshape(vec![batch_size, seq_len, embed_dim])?;

        let q_heads = self.split_heads(&q_seq, batch_size, seq_len)?;
        let k_heads = self.split_heads(&k_seq, batch_size, seq_len)?;
        let v_heads = self.split_heads(&v_seq, batch_size, seq_len)?;

        let scores = self.compute_attention_scores(&q_heads, &k_heads, batch_size, seq_len)?;
        let scale = 1.0f32 / (self.head_dim as f32).sqrt();
        let scaled_scores = scores.mul_scalar(scale)?;

        let probs_flat = scaled_scores
            .reshape(vec![batch_size * self.num_heads * seq_len, seq_len])?
            .softmax()?;
        let probs_4d =
            probs_flat
                .clone()
                .reshape(vec![batch_size, self.num_heads, seq_len, seq_len])?;

        let context_heads = self.apply_attention(&probs_4d, &v_heads, batch_size, seq_len)?;
        let context = self.merge_heads(&context_heads, batch_size, seq_len)?;

        let context_flat = context.reshape(vec![batch_size * seq_len, embed_dim])?;
        let mut output_flat = context_flat.matmul(&self.w_o)?;
        if let Some(bias) = &self.b_o {
            output_flat = output_flat.add(bias)?;
        }
        let output = output_flat.reshape(vec![batch_size, seq_len, embed_dim])?;

        let cache = MultiHeadAttentionCache {
            input_shape,
            batch_size,
            seq_len,
            flattened_input: flattened,
            q_heads,
            k_heads,
            v_heads,
            probs_4d,
            probs_flat,
            context_flat,
        };

        Ok((output, cache))
    }

    pub fn backward(
        &self,
        cache: &MultiHeadAttentionCache,
        grad_output: &Tensor,
    ) -> Result<(Tensor, MultiHeadAttentionGradients)> {
        if grad_output.shape() != cache.input_shape.as_slice() {
            return Err(anyhow!(
                "MultiHeadAttention backward expects grad_output with shape {:?}, got {:?}",
                cache.input_shape,
                grad_output.shape()
            ));
        }

        let grad_output_flat = grad_output
            .clone()
            .reshape(vec![cache.batch_size * cache.seq_len, self.embed_dim])?;

        let (grad_context_flat, grad_w_o) =
            Tensor::matmul_backward(&cache.context_flat, &self.w_o, &grad_output_flat)?;

        let grad_b_o = if self.b_o.is_some() {
            Some(grad_output_flat.sum_columns()?)
        } else {
            None
        };

        let grad_context =
            grad_context_flat.reshape(vec![cache.batch_size, cache.seq_len, self.embed_dim])?;
        let grad_context_heads =
            self.split_heads(&grad_context, cache.batch_size, cache.seq_len)?;

        let (grad_probs, grad_v_heads) = self.launch_apply_attention_backward(
            &grad_context_heads,
            &cache.probs_4d,
            &cache.v_heads,
            cache.batch_size,
            cache.seq_len,
        )?;

        let grad_probs_flat = grad_probs.reshape(vec![
            cache.batch_size * self.num_heads * cache.seq_len,
            cache.seq_len,
        ])?;
        let grad_scaled_flat = Tensor::softmax_backward(&cache.probs_flat, &grad_probs_flat)?;
        let grad_scaled = grad_scaled_flat.reshape(vec![
            cache.batch_size,
            self.num_heads,
            cache.seq_len,
            cache.seq_len,
        ])?;
        let scale = 1.0f32 / (self.head_dim as f32).sqrt();
        let grad_scores = grad_scaled.mul_scalar(scale)?;

        let (grad_q_heads, grad_k_heads) = self.launch_attention_scores_backward(
            &grad_scores,
            &cache.q_heads,
            &cache.k_heads,
            cache.batch_size,
            cache.seq_len,
        )?;

        let grad_v = self.merge_heads(&grad_v_heads, cache.batch_size, cache.seq_len)?;
        let grad_q = self.merge_heads(&grad_q_heads, cache.batch_size, cache.seq_len)?;
        let grad_k = self.merge_heads(&grad_k_heads, cache.batch_size, cache.seq_len)?;

        let grad_v_flat = grad_v.reshape(vec![cache.batch_size * cache.seq_len, self.embed_dim])?;
        let grad_q_flat = grad_q.reshape(vec![cache.batch_size * cache.seq_len, self.embed_dim])?;
        let grad_k_flat = grad_k.reshape(vec![cache.batch_size * cache.seq_len, self.embed_dim])?;

        let (grad_flat_q, grad_w_q) =
            Tensor::matmul_backward(&cache.flattened_input, &self.w_q, &grad_q_flat)?;
        let (grad_flat_k, grad_w_k) =
            Tensor::matmul_backward(&cache.flattened_input, &self.w_k, &grad_k_flat)?;
        let (grad_flat_v, grad_w_v) =
            Tensor::matmul_backward(&cache.flattened_input, &self.w_v, &grad_v_flat)?;

        let grad_b_q = if self.b_q.is_some() {
            Some(grad_q_flat.sum_columns()?)
        } else {
            None
        };
        let grad_b_k = if self.b_k.is_some() {
            Some(grad_k_flat.sum_columns()?)
        } else {
            None
        };
        let grad_b_v = if self.b_v.is_some() {
            Some(grad_v_flat.sum_columns()?)
        } else {
            None
        };

        let grad_input_flat = grad_flat_q.add(&grad_flat_k)?.add(&grad_flat_v)?;
        let grad_input = grad_input_flat.reshape(cache.input_shape.clone())?;

        let gradients = MultiHeadAttentionGradients {
            w_q: grad_w_q,
            w_k: grad_w_k,
            w_v: grad_w_v,
            w_o: grad_w_o,
            b_q: grad_b_q,
            b_k: grad_b_k,
            b_v: grad_b_v,
            b_o: grad_b_o,
        };

        Ok((grad_input, gradients))
    }

    fn launch_apply_attention_backward(
        &self,
        grad_context: &Tensor,
        attention: &Tensor,
        value: &Tensor,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<(Tensor, Tensor)> {
        let expected_context_shape = [batch_size, self.num_heads, seq_len, self.head_dim];
        if grad_context.shape() != expected_context_shape {
            return Err(anyhow!(
                "apply_attention_backward expects grad_context with shape {:?}, got {:?}",
                expected_context_shape,
                grad_context.shape()
            ));
        }
        let expected_attention_shape = [batch_size, self.num_heads, seq_len, seq_len];
        if attention.shape() != expected_attention_shape {
            return Err(anyhow!(
                "apply_attention_backward expects attention with shape {:?}, got {:?}",
                expected_attention_shape,
                attention.shape()
            ));
        }
        if value.shape() != expected_context_shape {
            return Err(anyhow!(
                "apply_attention_backward expects value with shape {:?}, got {:?}",
                expected_context_shape,
                value.shape()
            ));
        }

        if grad_context.device.as_raw() != attention.device.as_raw()
            || grad_context.device.as_raw() != value.device.as_raw()
        {
            return Err(anyhow!(
                "apply_attention_backward tensors must reside on the same CUDA device"
            ));
        }

        let grad_attention = Tensor::new(expected_attention_shape.to_vec(), &grad_context.device)?;
        let grad_value = Tensor::new(expected_context_shape.to_vec(), &grad_context.device)?;

        let total = batch_size
            .checked_mul(self.num_heads)
            .and_then(|v| v.checked_mul(seq_len))
            .and_then(|v| v.checked_mul(self.head_dim))
            .ok_or_else(|| anyhow!("apply_attention_backward launch configuration overflow"))?;
        if total > i32::MAX as usize {
            return Err(anyhow!(
                "apply_attention_backward tensor too large for kernel launch"
            ));
        }

        let block_size = 256u32;
        let grid_size = ((total as u32) + block_size - 1) / block_size;
        let stream =
            Stream::new(StreamFlags::NON_BLOCKING, None).context("Failed to create CUDA stream")?;
        let module = kernel_cache::module(
            include_str!("apply_attention_backward_kernel.ptx"),
            "apply_attention_backward",
        )?;
        let function = module
            .get_function("apply_attention_backward_kernel")
            .context("Kernel load failed for apply_attention_backward_kernel")?;

        unsafe {
            launch!(function<<<grid_size, block_size, 0, stream>>>(
                grad_context.data.as_device_ptr(),
                attention.data.as_device_ptr(),
                value.data.as_device_ptr(),
                grad_attention.data.as_device_ptr(),
                grad_value.data.as_device_ptr(),
                batch_size as i32,
                self.num_heads as i32,
                seq_len as i32,
                self.head_dim as i32
            ))?;
        }

        stream
            .synchronize()
            .context("Stream sync failed for apply_attention_backward_kernel")?;

        Ok((grad_attention, grad_value))
    }

    fn launch_attention_scores_backward(
        &self,
        grad_scores: &Tensor,
        query: &Tensor,
        key: &Tensor,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<(Tensor, Tensor)> {
        let expected_scores_shape = [batch_size, self.num_heads, seq_len, seq_len];
        if grad_scores.shape() != expected_scores_shape {
            return Err(anyhow!(
                "attention_scores_backward expects grad_scores with shape {:?}, got {:?}",
                expected_scores_shape,
                grad_scores.shape()
            ));
        }
        let expected_proj_shape = [batch_size, self.num_heads, seq_len, self.head_dim];
        for (name, tensor) in [("query", query), ("key", key)] {
            if tensor.shape() != expected_proj_shape {
                return Err(anyhow!(
                    "attention_scores_backward expects {name} with shape {:?}, got {:?}",
                    expected_proj_shape,
                    tensor.shape()
                ));
            }
        }

        if grad_scores.device.as_raw() != query.device.as_raw()
            || grad_scores.device.as_raw() != key.device.as_raw()
        {
            return Err(anyhow!(
                "attention_scores_backward tensors must reside on the same CUDA device"
            ));
        }

        let grad_query = Tensor::new(expected_proj_shape.to_vec(), &grad_scores.device)?;
        let grad_key = Tensor::new(expected_proj_shape.to_vec(), &grad_scores.device)?;

        let total = batch_size
            .checked_mul(self.num_heads)
            .and_then(|v| v.checked_mul(seq_len))
            .and_then(|v| v.checked_mul(seq_len))
            .ok_or_else(|| anyhow!("attention_scores_backward launch configuration overflow"))?;
        if total > i32::MAX as usize {
            return Err(anyhow!(
                "attention_scores_backward tensor too large for kernel launch"
            ));
        }

        let block_size = 256u32;
        let grid_size = ((total as u32) + block_size - 1) / block_size;
        let stream =
            Stream::new(StreamFlags::NON_BLOCKING, None).context("Failed to create CUDA stream")?;
        let module = kernel_cache::module(
            include_str!("attention_scores_backward_kernel.ptx"),
            "attention_scores_backward",
        )?;
        let function = module
            .get_function("attention_scores_backward_kernel")
            .context("Kernel load failed for attention_scores_backward_kernel")?;

        unsafe {
            launch!(function<<<grid_size, block_size, 0, stream>>>(
                grad_scores.data.as_device_ptr(),
                query.data.as_device_ptr(),
                key.data.as_device_ptr(),
                grad_query.data.as_device_ptr(),
                grad_key.data.as_device_ptr(),
                batch_size as i32,
                self.num_heads as i32,
                seq_len as i32,
                self.head_dim as i32
            ))?;
        }

        stream
            .synchronize()
            .context("Stream sync failed for attention_scores_backward_kernel")?;

        Ok((grad_query, grad_key))
    }

    pub fn append_named_tensors<'a>(&'a self, prefix: &str, out: &mut Vec<(String, &'a Tensor)>) {
        out.push((format!("{prefix}.w_q"), &self.w_q));
        out.push((format!("{prefix}.w_k"), &self.w_k));
        out.push((format!("{prefix}.w_v"), &self.w_v));
        out.push((format!("{prefix}.w_o"), &self.w_o));

        if let Some(bias) = &self.b_q {
            out.push((format!("{prefix}.b_q"), bias));
        }
        if let Some(bias) = &self.b_k {
            out.push((format!("{prefix}.b_k"), bias));
        }
        if let Some(bias) = &self.b_v {
            out.push((format!("{prefix}.b_v"), bias));
        }
        if let Some(bias) = &self.b_o {
            out.push((format!("{prefix}.b_o"), bias));
        }
    }

    pub fn visit_parameters_mut<F>(&mut self, prefix: &str, f: &mut F) -> Result<()>
    where
        F: FnMut(&str, &mut Tensor) -> Result<()>,
    {
        f(&format!("{prefix}.w_q"), &mut self.w_q)?;
        f(&format!("{prefix}.w_k"), &mut self.w_k)?;
        f(&format!("{prefix}.w_v"), &mut self.w_v)?;
        f(&format!("{prefix}.w_o"), &mut self.w_o)?;

        if let Some(bias) = &mut self.b_q {
            f(&format!("{prefix}.b_q"), bias)?;
        }
        if let Some(bias) = &mut self.b_k {
            f(&format!("{prefix}.b_k"), bias)?;
        }
        if let Some(bias) = &mut self.b_v {
            f(&format!("{prefix}.b_v"), bias)?;
        }
        if let Some(bias) = &mut self.b_o {
            f(&format!("{prefix}.b_o"), bias)?;
        }

        Ok(())
    }
}

impl Layer for MultiHeadAttention {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (output, _) = self.forward_with_cache(input)?;
        Ok(output)
    }
}

pub struct FeedForwardNetwork {
    input_dim: usize,
    hidden_dim: usize,
    w1: Tensor,
    b1: Option<Tensor>,
    w2: Tensor,
    b2: Option<Tensor>,
    activation: ActivationKind,
}

#[allow(dead_code)]
impl FeedForwardNetwork {
    pub fn new(
        device: &Device,
        input_dim: usize,
        hidden_dim: usize,
        activation: ActivationKind,
    ) -> Result<Self> {
        let w1_scale = 1.0f32 / (input_dim as f32).sqrt();
        let w2_scale = 1.0f32 / (hidden_dim as f32).sqrt();
        let w1 = Tensor::randn(vec![input_dim, hidden_dim], device)?;
        let w1 = w1.mul_scalar(w1_scale)?;
        let w2 = Tensor::randn(vec![hidden_dim, input_dim], device)?;
        let w2 = w2.mul_scalar(w2_scale)?;
        let b1 = Tensor::new(vec![1, hidden_dim], device)?;
        let b2 = Tensor::new(vec![1, input_dim], device)?;
        Self::assemble(
            input_dim,
            hidden_dim,
            w1,
            Some(b1),
            w2,
            Some(b2),
            activation,
        )
    }

    pub fn new_without_bias(
        device: &Device,
        input_dim: usize,
        hidden_dim: usize,
        activation: ActivationKind,
    ) -> Result<Self> {
        let w1_scale = 1.0f32 / (input_dim as f32).sqrt();
        let w2_scale = 1.0f32 / (hidden_dim as f32).sqrt();
        let w1 = Tensor::randn(vec![input_dim, hidden_dim], device)?;
        let w1 = w1.mul_scalar(w1_scale)?;
        let w2 = Tensor::randn(vec![hidden_dim, input_dim], device)?;
        let w2 = w2.mul_scalar(w2_scale)?;
        Self::assemble(input_dim, hidden_dim, w1, None, w2, None, activation)
    }

    pub fn from_host(
        device: &Device,
        input_dim: usize,
        hidden_dim: usize,
        w1: Vec<f32>,
        b1: Option<Vec<f32>>,
        w2: Vec<f32>,
        b2: Option<Vec<f32>>,
        activation: ActivationKind,
    ) -> Result<Self> {
        let w1 = Tensor::from_host(w1, vec![input_dim, hidden_dim], device)?;
        let w2 = Tensor::from_host(w2, vec![hidden_dim, input_dim], device)?;
        let b1 = match b1 {
            Some(data) => Some(Tensor::from_host(data, vec![1, hidden_dim], device)?),
            None => None,
        };
        let b2 = match b2 {
            Some(data) => Some(Tensor::from_host(data, vec![1, input_dim], device)?),
            None => None,
        };

        Self::assemble(input_dim, hidden_dim, w1, b1, w2, b2, activation)
    }

    fn assemble(
        input_dim: usize,
        hidden_dim: usize,
        w1: Tensor,
        b1: Option<Tensor>,
        w2: Tensor,
        b2: Option<Tensor>,
        activation: ActivationKind,
    ) -> Result<Self> {
        if input_dim == 0 || hidden_dim == 0 {
            return Err(anyhow!(
                "FeedForwardNetwork dimensions must be greater than zero"
            ));
        }

        if w1.shape() != &[input_dim, hidden_dim] {
            return Err(anyhow!(
                "w1 must have shape [{}, {}], found {:?}",
                input_dim,
                hidden_dim,
                w1.shape()
            ));
        }
        if w2.shape() != &[hidden_dim, input_dim] {
            return Err(anyhow!(
                "w2 must have shape [{}, {}], found {:?}",
                hidden_dim,
                input_dim,
                w2.shape()
            ));
        }

        let device_raw = w1.device.as_raw();
        if w2.device.as_raw() != device_raw {
            return Err(anyhow!(
                "FeedForwardNetwork weights must share the same device"
            ));
        }

        for (bias, expected) in [(b1.as_ref(), hidden_dim), (b2.as_ref(), input_dim)] {
            if let Some(bias) = bias {
                if bias.device.as_raw() != device_raw {
                    return Err(anyhow!(
                        "FeedForwardNetwork biases must share the same device"
                    ));
                }
                if bias.shape() != &[1, expected] {
                    return Err(anyhow!(
                        "Bias must have shape [1, {}], found {:?}",
                        expected,
                        bias.shape()
                    ));
                }
            }
        }

        Ok(Self {
            input_dim,
            hidden_dim,
            w1,
            b1,
            w2,
            b2,
            activation,
        })
    }

    pub fn append_named_tensors<'a>(&'a self, prefix: &str, out: &mut Vec<(String, &'a Tensor)>) {
        out.push((format!("{prefix}.w1"), &self.w1));
        out.push((format!("{prefix}.w2"), &self.w2));
        if let Some(bias) = &self.b1 {
            out.push((format!("{prefix}.b1"), bias));
        }
        if let Some(bias) = &self.b2 {
            out.push((format!("{prefix}.b2"), bias));
        }
    }

    pub fn visit_parameters_mut<F>(&mut self, prefix: &str, f: &mut F) -> Result<()>
    where
        F: FnMut(&str, &mut Tensor) -> Result<()>,
    {
        f(&format!("{prefix}.w1"), &mut self.w1)?;
        f(&format!("{prefix}.w2"), &mut self.w2)?;
        if let Some(bias) = &mut self.b1 {
            f(&format!("{prefix}.b1"), bias)?;
        }
        if let Some(bias) = &mut self.b2 {
            f(&format!("{prefix}.b2"), bias)?;
        }
        Ok(())
    }
}

impl Layer for FeedForwardNetwork {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if input.shape().is_empty() {
            return Err(anyhow!(
                "FeedForwardNetwork expects input with at least one dimension, got {:?}",
                input.shape()
            ));
        }

        if *input.shape().last().unwrap() != self.input_dim {
            return Err(anyhow!(
                "FeedForwardNetwork expects last dimension {} but found {}",
                self.input_dim,
                input.shape().last().unwrap()
            ));
        }

        if input.device.as_raw() != self.w1.device.as_raw() {
            return Err(anyhow!(
                "FeedForwardNetwork input tensor must share the same device as the weights"
            ));
        }

        let rows = input.size() / self.input_dim;
        if rows == 0 {
            return Err(anyhow!(
                "FeedForwardNetwork cannot process an empty input tensor"
            ));
        }

        let flat = input.clone().reshape(vec![rows, self.input_dim])?;

        let mut hidden = flat.matmul(&self.w1)?;
        if let Some(bias) = &self.b1 {
            hidden = hidden.add(bias)?;
        }

        debug_assert_eq!(hidden.shape(), &[rows, self.hidden_dim]);

        hidden = match self.activation {
            ActivationKind::Relu => hidden.relu()?,
        };

        let mut output = hidden.matmul(&self.w2)?;
        if let Some(bias) = &self.b2 {
            output = output.add(bias)?;
        }

        debug_assert_eq!(output.shape(), &[rows, self.input_dim]);

        output.reshape(input.shape().to_vec())
    }
}

pub struct FeedForwardCache {
    input_shape: Vec<usize>,
    flat_input: Tensor,
    hidden_pre_activation: Tensor,
    hidden_post_activation: Tensor,
}

pub struct FeedForwardGradients {
    pub w1: Tensor,
    pub b1: Option<Tensor>,
    pub w2: Tensor,
    pub b2: Option<Tensor>,
}

impl FeedForwardNetwork {
    pub fn forward_with_cache(&self, input: &Tensor) -> Result<(Tensor, FeedForwardCache)> {
        if input.shape().is_empty() {
            return Err(anyhow!(
                "FeedForwardNetwork expects input with at least one dimension, got {:?}",
                input.shape()
            ));
        }

        if *input.shape().last().unwrap() != self.input_dim {
            return Err(anyhow!(
                "FeedForwardNetwork expects last dimension {} but found {}",
                self.input_dim,
                input.shape().last().unwrap()
            ));
        }

        if input.device.as_raw() != self.w1.device.as_raw() {
            return Err(anyhow!(
                "FeedForwardNetwork input tensor must share the same device as the weights"
            ));
        }

        let rows = input.size() / self.input_dim;
        let flat_input = input.clone().reshape(vec![rows, self.input_dim])?;

        let mut hidden_pre = flat_input.matmul(&self.w1)?;
        if let Some(bias) = &self.b1 {
            hidden_pre = hidden_pre.add(bias)?;
        }
        let hidden_post = hidden_pre.relu()?;

        let mut output_flat = hidden_post.matmul(&self.w2)?;
        if let Some(bias) = &self.b2 {
            output_flat = output_flat.add(bias)?;
        }

        let output = output_flat.clone().reshape(input.shape().to_vec())?;
        let cache = FeedForwardCache {
            input_shape: input.shape().to_vec(),
            flat_input,
            hidden_pre_activation: hidden_pre,
            hidden_post_activation: hidden_post,
        };

        Ok((output, cache))
    }

    pub fn backward(
        &self,
        cache: &FeedForwardCache,
        grad_output: &Tensor,
    ) -> Result<(Tensor, FeedForwardGradients)> {
        if grad_output.shape() != cache.input_shape.as_slice() {
            return Err(anyhow!(
                "FeedForwardNetwork backward expects grad_output with shape {:?}, got {:?}",
                cache.input_shape,
                grad_output.shape()
            ));
        }

        let rows = cache.flat_input.shape()[0];
        let grad_output_flat = grad_output.clone().reshape(vec![rows, self.input_dim])?;

        let (grad_hidden_post, grad_w2) =
            Tensor::matmul_backward(&cache.hidden_post_activation, &self.w2, &grad_output_flat)?;

        let grad_b2 = if self.b2.is_some() {
            Some(grad_output_flat.sum_columns()?)
        } else {
            None
        };

        let grad_hidden_pre =
            Tensor::relu_backward(&cache.hidden_pre_activation, &grad_hidden_post)?;

        let (grad_flat_input, grad_w1) =
            Tensor::matmul_backward(&cache.flat_input, &self.w1, &grad_hidden_pre)?;

        let grad_b1 = if self.b1.is_some() {
            Some(grad_hidden_pre.sum_columns()?)
        } else {
            None
        };

        let grad_input = grad_flat_input.reshape(cache.input_shape.clone())?;

        let gradients = FeedForwardGradients {
            w1: grad_w1,
            b1: grad_b1,
            w2: grad_w2,
            b2: grad_b2,
        };

        Ok((grad_input, gradients))
    }
}

pub struct LayerNorm {
    normalized_dim: usize,
    gamma: Tensor,
    beta: Tensor,
    eps: f32,
}

#[allow(dead_code)]
impl LayerNorm {
    pub fn new(device: &Device, normalized_dim: usize, eps: f32) -> Result<Self> {
        let gamma = Tensor::ones(vec![normalized_dim], device)?;
        let beta = Tensor::new(vec![normalized_dim], device)?;
        Self::assemble(normalized_dim, gamma, beta, eps)
    }

    pub fn from_host(
        device: &Device,
        normalized_dim: usize,
        gamma: Option<Vec<f32>>,
        beta: Option<Vec<f32>>,
        eps: f32,
    ) -> Result<Self> {
        let gamma = match gamma {
            Some(values) => Tensor::from_host(values, vec![normalized_dim], device)?,
            None => Tensor::ones(vec![normalized_dim], device)?,
        };
        let beta = match beta {
            Some(values) => Tensor::from_host(values, vec![normalized_dim], device)?,
            None => Tensor::new(vec![normalized_dim], device)?,
        };

        Self::assemble(normalized_dim, gamma, beta, eps)
    }

    fn assemble(normalized_dim: usize, gamma: Tensor, beta: Tensor, eps: f32) -> Result<Self> {
        if normalized_dim == 0 {
            return Err(anyhow!(
                "LayerNorm normalized dimension must be greater than zero"
            ));
        }

        if gamma.shape() != &[normalized_dim] {
            return Err(anyhow!(
                "gamma must have shape [{}], found {:?}",
                normalized_dim,
                gamma.shape()
            ));
        }

        if beta.shape() != &[normalized_dim] {
            return Err(anyhow!(
                "beta must have shape [{}], found {:?}",
                normalized_dim,
                beta.shape()
            ));
        }

        if gamma.device.as_raw() != beta.device.as_raw() {
            return Err(anyhow!(
                "LayerNorm gamma and beta must live on the same device"
            ));
        }

        Ok(Self {
            normalized_dim,
            gamma,
            beta,
            eps,
        })
    }

    pub fn append_named_tensors<'a>(&'a self, prefix: &str, out: &mut Vec<(String, &'a Tensor)>) {
        out.push((format!("{prefix}.gamma"), &self.gamma));
        out.push((format!("{prefix}.beta"), &self.beta));
    }

    pub fn visit_parameters_mut<F>(&mut self, prefix: &str, f: &mut F) -> Result<()>
    where
        F: FnMut(&str, &mut Tensor) -> Result<()>,
    {
        f(&format!("{prefix}.gamma"), &mut self.gamma)?;
        f(&format!("{prefix}.beta"), &mut self.beta)?;
        Ok(())
    }
}

impl Layer for LayerNorm {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if input.shape().is_empty() {
            return Err(anyhow!(
                "LayerNorm expects a tensor with at least one dimension"
            ));
        }

        let last_dim = *input.shape().last().unwrap();
        if last_dim != self.normalized_dim {
            return Err(anyhow!(
                "LayerNorm expected last dimension {} but found {}",
                self.normalized_dim,
                last_dim
            ));
        }

        if input.device.as_raw() != self.gamma.device.as_raw() {
            return Err(anyhow!(
                "LayerNorm input and parameters must share the same device"
            ));
        }

        let rows = input.size() / self.normalized_dim;
        if rows == 0 {
            return Err(anyhow!("LayerNorm cannot normalize an empty tensor"));
        }

        let rows_i32 = i32::try_from(rows)
            .map_err(|_| anyhow!("LayerNorm input too large for kernel launch"))?;
        let cols_i32 = i32::try_from(self.normalized_dim)
            .map_err(|_| anyhow!("LayerNorm normalized dimension too large for kernel launch"))?;

        let mut block_size = 256u32;
        while block_size > 1 && block_size > self.normalized_dim as u32 {
            block_size /= 2;
        }
        if block_size == 0 {
            block_size = 1;
        }

        let grid_size = u32::try_from(rows)
            .map_err(|_| anyhow!("LayerNorm row count exceeds CUDA grid limits"))?;
        let shared_bytes = (2 * block_size as usize * std::mem::size_of::<f32>()) as u32;

        let result = Tensor::new(input.shape().to_vec(), &input.device)?;
        let stream =
            Stream::new(StreamFlags::NON_BLOCKING, None).context("Failed to create CUDA stream")?;
        let module = kernel_cache::module(include_str!("layer_norm_kernel.ptx"), "layer_norm")?;
        let function = module
            .get_function("layer_norm_kernel")
            .context("Kernel load failed for layer_norm")?;

        unsafe {
            launch!(function<<<grid_size, block_size, shared_bytes, stream>>>(
                input.data.as_device_ptr(),
                self.gamma.data.as_device_ptr(),
                self.beta.data.as_device_ptr(),
                result.data.as_device_ptr(),
                rows_i32,
                cols_i32,
                self.eps
            ))?;
        }

        stream
            .synchronize()
            .context("Stream sync failed for layer_norm")?;

        Ok(result)
    }
}

pub struct LayerNormCache {
    input_shape: Vec<usize>,
    rows: usize,
    normalized: Tensor,
    inv_std: Tensor,
}

pub struct LayerNormGradients {
    pub gamma: Tensor,
    pub beta: Tensor,
}

impl LayerNorm {
    pub fn forward_with_cache(&self, input: &Tensor) -> Result<(Tensor, LayerNormCache)> {
        if input.shape().is_empty() {
            return Err(anyhow!(
                "LayerNorm expects a tensor with at least one dimension"
            ));
        }

        let last_dim = *input.shape().last().unwrap();
        if last_dim != self.normalized_dim {
            return Err(anyhow!(
                "LayerNorm expected last dimension {} but found {}",
                self.normalized_dim,
                last_dim
            ));
        }

        if input.device.as_raw() != self.gamma.device.as_raw() {
            return Err(anyhow!(
                "LayerNorm input and parameters must share the same device"
            ));
        }

        let input_shape = input.shape().to_vec();
        let rows = input.size() / self.normalized_dim;
        let input_2d = input.clone().reshape(vec![rows, self.normalized_dim])?;

        let sum = input_2d.sum_rows()?;
        let mean = sum.div_scalar(self.normalized_dim as f32)?;
        let mean_2d = mean.reshape(vec![rows, 1])?;
        let centered = input_2d.sub(&mean_2d)?;

        let squared = centered.mul(&centered)?;
        let variance = squared.sum_rows()?.div_scalar(self.normalized_dim as f32)?;
        let variance_eps = variance.add_scalar(self.eps)?;

        let variance_host = variance_eps.to_host()?;
        let inv_std_host: Vec<f32> = variance_host
            .iter()
            .map(|value| 1.0f32 / value.sqrt())
            .collect();
        let inv_std_2d = Tensor::from_host(inv_std_host.clone(), vec![rows, 1], &input.device)?;
        let inv_std = Tensor::from_host(inv_std_host, vec![rows], &input.device)?;

        let normalized = centered.mul(&inv_std_2d)?;
        let gamma_broadcast = self.gamma.clone().reshape(vec![1, self.normalized_dim])?;
        let beta_broadcast = self.beta.clone().reshape(vec![1, self.normalized_dim])?;
        let output_2d = normalized.mul(&gamma_broadcast)?.add(&beta_broadcast)?;
        let output = output_2d.clone().reshape(input_shape.clone())?;

        let cache = LayerNormCache {
            input_shape,
            rows,
            normalized,
            inv_std,
        };

        Ok((output, cache))
    }

    pub fn backward(
        &self,
        cache: &LayerNormCache,
        grad_output: &Tensor,
    ) -> Result<(Tensor, LayerNormGradients)> {
        if grad_output.shape() != cache.input_shape.as_slice() {
            return Err(anyhow!(
                "LayerNorm backward expects grad_output with shape {:?}, got {:?}",
                cache.input_shape,
                grad_output.shape()
            ));
        }

        let grad_output_2d = grad_output
            .clone()
            .reshape(vec![cache.rows, self.normalized_dim])?;
        let normalized = cache.normalized.clone();
        let inv_std_2d = cache.inv_std.clone().reshape(vec![cache.rows, 1])?;

        let gamma_broadcast = self.gamma.clone().reshape(vec![1, self.normalized_dim])?;
        let grad_scaled = grad_output_2d.mul(&gamma_broadcast)?;

        let sum_grad = grad_scaled
            .clone()
            .sum_rows()?
            .reshape(vec![cache.rows, 1])?;
        let sum_dot = grad_scaled
            .clone()
            .mul(&normalized)?
            .sum_rows()?
            .reshape(vec![cache.rows, 1])?;

        let n = self.normalized_dim as f32;
        let numerator = grad_scaled
            .mul_scalar(n)?
            .sub(&sum_grad)?
            .sub(&normalized.mul(&sum_dot)?)?;
        let grad_input_2d = numerator.mul(&inv_std_2d)?.mul_scalar(1.0 / n)?;
        let grad_input = grad_input_2d.reshape(cache.input_shape.clone())?;

        let grad_gamma = grad_output_2d
            .clone()
            .mul(&normalized)?
            .transpose2d()?
            .sum_rows()?;
        let grad_gamma = grad_gamma.reshape(vec![self.normalized_dim])?;

        let grad_beta = grad_output_2d.transpose2d()?.sum_rows()?;
        let grad_beta = grad_beta.reshape(vec![self.normalized_dim])?;

        let gradients = LayerNormGradients {
            gamma: grad_gamma,
            beta: grad_beta,
        };

        Ok((grad_input, gradients))
    }
}
