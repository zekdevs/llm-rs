use anyhow::{Context, Result, anyhow};
use cust::device::Device;
use cust::module::Module;
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::convert::TryFrom;
use serde::{Deserialize, Serialize};

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

        let w_q = Tensor::randn(vec![embed_dim, embed_dim], device)?;
        let w_k = Tensor::randn(vec![embed_dim, embed_dim], device)?;
        let w_v = Tensor::randn(vec![embed_dim, embed_dim], device)?;
        let w_o = Tensor::randn(vec![embed_dim, embed_dim], device)?;

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
        let module = Module::from_ptx(include_str!("split_heads_kernel.ptx"), &[])
            .context("PTX load failed for split_heads")?;
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
        let module = Module::from_ptx(include_str!("merge_heads_kernel.ptx"), &[])
            .context("PTX load failed for merge_heads")?;
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
        let module = Module::from_ptx(include_str!("attention_scores_kernel.ptx"), &[])
            .context("PTX load failed for attention_scores")?;
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
        let module = Module::from_ptx(include_str!("apply_attention_kernel.ptx"), &[])
            .context("PTX load failed for apply_attention")?;
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
}

impl Layer for MultiHeadAttention {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
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

        let flattened = input
            .clone()
            .reshape(vec![batch_size * seq_len, embed_dim])?;

        let q = self.project(&flattened, &self.w_q, self.b_q.as_ref())?;
        let k = self.project(&flattened, &self.w_k, self.b_k.as_ref())?;
        let v = self.project(&flattened, &self.w_v, self.b_v.as_ref())?;

        let q = q.reshape(vec![batch_size, seq_len, embed_dim])?;
        let k = k.reshape(vec![batch_size, seq_len, embed_dim])?;
        let v = v.reshape(vec![batch_size, seq_len, embed_dim])?;

        let q_heads = self.split_heads(&q, batch_size, seq_len)?;
        let k_heads = self.split_heads(&k, batch_size, seq_len)?;
        let v_heads = self.split_heads(&v, batch_size, seq_len)?;

        let scores = self.compute_attention_scores(&q_heads, &k_heads, batch_size, seq_len)?;
        let scale = 1.0f32 / (self.head_dim as f32).sqrt();
        let scaled_scores = scores.mul_scalar(scale)?;

        let probs = scaled_scores
            .reshape(vec![batch_size * self.num_heads * seq_len, seq_len])?
            .softmax()?
            .reshape(vec![batch_size, self.num_heads, seq_len, seq_len])?;

        let context = self.apply_attention(&probs, &v_heads, batch_size, seq_len)?;
        let context = self.merge_heads(&context, batch_size, seq_len)?;

        let output = context
            .reshape(vec![batch_size * seq_len, embed_dim])?
            .matmul(&self.w_o)?;

        let output = if let Some(bias) = &self.b_o {
            output.add(bias)?
        } else {
            output
        };

        output.reshape(vec![batch_size, seq_len, embed_dim])
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
        let w1 = Tensor::randn(vec![input_dim, hidden_dim], device)?;
        let w2 = Tensor::randn(vec![hidden_dim, input_dim], device)?;
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
        let w1 = Tensor::randn(vec![input_dim, hidden_dim], device)?;
        let w2 = Tensor::randn(vec![hidden_dim, input_dim], device)?;
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
        let module = Module::from_ptx(include_str!("layer_norm_kernel.ptx"), &[])
            .context("PTX load failed for layer_norm")?;
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
